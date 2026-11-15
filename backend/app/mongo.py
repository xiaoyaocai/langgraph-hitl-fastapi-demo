"""MongoDB client and LangGraph short-term memory checkpointer."""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
)
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.mongodb.saver import WRITES_IDX_MAP
from pymongo import MongoClient, UpdateOne
from typing_extensions import override

DEFAULT_MONGO_URI = (
    "mongodb://langgraph:langgraph@192.168.184.144:27017/rag_db?authSource=admin"
)
DEFAULT_DB = "rag_db"
DEFAULT_CHECKPOINT_COLLECTION = "short_term_memory"
DEFAULT_WRITES_COLLECTION = "short_term_memory_writes"
DEFAULT_LONG_TERM_COLLECTION = "long_term_memory"

_MESSAGE_ROLE_ZH = {
    "human": "用户",
    "ai": "助手",
    "system": "系统",
    "tool": "工具",
    "function": "函数",
}

_STATUS_ZH = {
    "approved": "已批准",
    "feedback": "待反馈",
    "finished": "已完成",
    "pending": "处理中",
    "user_feedback": "等待人工反馈",
}

_SOURCE_ZH = {
    "input": "输入",
    "loop": "循环",
    "update": "更新",
    "fork": "分叉",
}

_CHANNEL_ZH = {
    "messages": "对话",
    "human_request": "用户请求",
    "assistant_response": "助手回复",
    "human_comment": "人工反馈",
    "status": "状态",
    "branch:to:assistant_draft": "路由到草稿节点",
    "branch:to:human_feedback": "路由到人工反馈",
    "branch:to:assistant_finalize": "路由到定稿节点",
    "__start__": "开始",
    "__error__": "错误",
    "__interrupt__": "中断",
    "__resume__": "恢复",
    "__scheduled__": "调度",
}

_client: Optional[MongoClient] = None
_checkpointer: Optional[MongoDBSaver] = None


def get_mongo_uri() -> str:
    return os.getenv("MONGO_URI") or os.getenv("MONGODB_URI") or DEFAULT_MONGO_URI


def get_mongo_db() -> str:
    return os.getenv("MONGO_DB", DEFAULT_DB)


def get_mongo_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(get_mongo_uri(), serverSelectionTimeoutMS=5000)
    return _client


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                parts.append(text if isinstance(text, str) else str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _serialize_message_zh(message: Any) -> dict[str, str]:
    if isinstance(message, BaseMessage):
        role = _MESSAGE_ROLE_ZH.get(message.type, message.type)
        return {"角色": role, "内容": _message_content_to_text(message.content)}
    if isinstance(message, dict):
        raw_role = str(message.get("type") or message.get("role") or "unknown")
        role = _MESSAGE_ROLE_ZH.get(raw_role, raw_role)
        return {
            "角色": role,
            "内容": _message_content_to_text(message.get("content")),
        }
    return {"角色": "未知", "内容": str(message)}


def build_chinese_io_payload(
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata | None = None,
) -> dict[str, Any]:
    """Build a Chinese, human-readable I/O projection from checkpoint state."""
    values = checkpoint.get("channel_values") or {}
    messages = values.get("messages") or []
    dialogue = [_serialize_message_zh(m) for m in messages if m is not None]

    human_request = values.get("human_request")
    assistant_response = values.get("assistant_response")
    human_comment = values.get("human_comment")
    status = values.get("status")

    # MCP / messages-only graphs: fall back to last human/ai turns.
    if human_request is None:
        for item in reversed(dialogue):
            if item.get("角色") == "用户":
                human_request = item.get("内容")
                break
    if assistant_response is None:
        for item in reversed(dialogue):
            if item.get("角色") == "助手":
                assistant_response = item.get("内容")
                break

    status_text = None
    if status is not None:
        status_key = str(status)
        status_text = _STATUS_ZH.get(status_key, status_key)

    meta = metadata or {}
    source = meta.get("source")
    source_text = _SOURCE_ZH.get(str(source), source) if source is not None else None

    return {
        "输入": human_request if human_request is not None else "",
        "输出": assistant_response if assistant_response is not None else "",
        "对话": dialogue,
        "状态": status_text,
        "人工反馈": human_comment if human_comment is not None else None,
        "来源": source_text,
        "步骤": meta.get("step"),
        "时间": checkpoint.get("ts"),
    }


def _channel_name_zh(channel: str) -> str:
    if channel in _CHANNEL_ZH:
        return _CHANNEL_ZH[channel]
    if channel.startswith("branch:to:"):
        target = channel.split(":", 2)[-1]
        return f"路由到{target}"
    return channel


def _task_node_zh(task_path: str) -> str | None:
    if not task_path:
        return None
    # e.g. "~__pregel_pull, assistant_draft"
    node = task_path.split(",")[-1].strip()
    mapping = {
        "assistant_draft": "草稿节点",
        "human_feedback": "人工反馈节点",
        "assistant_finalize": "定稿节点",
        "__start__": "开始节点",
    }
    return mapping.get(node, node)


def build_chinese_write_value(channel: str, value: Any) -> Any:
    """Convert a pending-write value into a Chinese-readable payload."""
    if value is None:
        return None

    if channel == "messages":
        if isinstance(value, list):
            return [_serialize_message_zh(m) for m in value if m is not None]
        return [_serialize_message_zh(value)]

    if channel == "status":
        key = str(value)
        return _STATUS_ZH.get(key, key)

    if channel in {"human_request", "assistant_response", "human_comment"}:
        return _message_content_to_text(value) if not isinstance(value, str) else value

    if isinstance(value, BaseMessage):
        return _serialize_message_zh(value)

    if isinstance(value, list) and value and isinstance(value[0], BaseMessage):
        return [_serialize_message_zh(m) for m in value]

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): build_chinese_write_value(channel, v) for k, v in value.items()}

    return _message_content_to_text(value) if hasattr(value, "content") else str(value)


def build_chinese_write_payload(
    channel: str,
    value: Any,
    task_path: str = "",
) -> dict[str, Any]:
    content = build_chinese_write_value(channel, value)
    payload = {
        "通道": _channel_name_zh(channel),
        "通道英文": channel,
        "内容": content,
        "节点": _task_node_zh(task_path),
    }
    return {
        "通道": payload["通道"],
        "通道英文": payload["通道英文"],
        "内容": payload["内容"],
        "节点": payload["节点"],
        "写入内容": payload,
    }


class ChineseIOMongoDBSaver(MongoDBSaver):
    """MongoDBSaver that also stores Chinese-readable input/output fields."""

    @override
    def put(
        self,
        config,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ):
        result = super().put(config, checkpoint, metadata, new_versions)
        conf = result["configurable"]
        chinese_io = build_chinese_io_payload(checkpoint, metadata)
        self.checkpoint_collection.update_one(
            {
                "thread_id": conf["thread_id"],
                "checkpoint_ns": conf["checkpoint_ns"],
                "checkpoint_id": conf["checkpoint_id"],
            },
            {
                "$set": {
                    "输入": chinese_io["输入"],
                    "输出": chinese_io["输出"],
                    "对话": chinese_io["对话"],
                    "状态": chinese_io["状态"],
                    "人工反馈": chinese_io["人工反馈"],
                    "来源": chinese_io["来源"],
                    "步骤": chinese_io["步骤"],
                    "时间": chinese_io["时间"],
                    "输入输出": chinese_io,
                }
            },
        )
        return result

    @override
    def put_writes(
        self,
        config,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        super().put_writes(config, writes, task_id, task_path)

        conf = config["configurable"]
        thread_id = conf["thread_id"]
        checkpoint_ns = conf["checkpoint_ns"]
        checkpoint_id = conf["checkpoint_id"]

        operations: list[UpdateOne] = []
        for idx, (channel, value) in enumerate(writes):
            write_idx = WRITES_IDX_MAP.get(channel, idx)
            chinese = build_chinese_write_payload(channel, value, task_path)
            operations.append(
                UpdateOne(
                    filter={
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                        "task_path": task_path,
                        "idx": write_idx,
                    },
                    update={"$set": chinese},
                )
            )
        if operations:
            self.writes_collection.bulk_write(operations)


def get_short_term_memory_checkpointer() -> MongoDBSaver:
    """
    Shared LangGraph checkpointer persisting session short-term memory
    to rag_db.short_term_memory (plus short_term_memory_writes for pending writes).

    Checkpoint docs store: 输入 / 输出 / 对话 / 状态 / 人工反馈.
    Write docs store: 通道 / 内容 / 节点 / 写入内容.
    """
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = ChineseIOMongoDBSaver(
            get_mongo_client(),
            db_name=get_mongo_db(),
            checkpoint_collection_name=os.getenv(
                "MONGO_SHORT_TERM_COLLECTION", DEFAULT_CHECKPOINT_COLLECTION
            ),
            writes_collection_name=os.getenv(
                "MONGO_SHORT_TERM_WRITES_COLLECTION", DEFAULT_WRITES_COLLECTION
            ),
        )
    return _checkpointer


def get_short_term_memory_collection():
    """Direct access to the short_term_memory collection (for inspection/ops)."""
    name = os.getenv(
        "MONGO_SHORT_TERM_COLLECTION", DEFAULT_CHECKPOINT_COLLECTION
    )
    return get_mongo_client()[get_mongo_db()][name]


def get_long_term_memory_collection():
    """Direct access to rag_db.long_term_memory."""
    name = os.getenv(
        "MONGO_LONG_TERM_COLLECTION", DEFAULT_LONG_TERM_COLLECTION
    )
    return get_mongo_client()[get_mongo_db()][name]
