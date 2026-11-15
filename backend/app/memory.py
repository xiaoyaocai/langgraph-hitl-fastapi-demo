"""Long-term memory manager: extract, persist, retrieve, delete via MongoDB."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Sequence
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.mongo import get_long_term_memory_collection

MEMORY_SYSTEM_PROMPT = """
你是一个专业的 AI Agent 记忆管理器。

你的任务是分析用户对话，提取值得保留的信息，
并判断信息应该存储为短期记忆还是长期记忆。

分类规则：

1. short_term：
   - 当前任务、当前问题
   - 临时参数、临时数据
   - 当前任务执行进度
   - 当前会话上下文

2. long_term：
   - 用户稳定的个人背景
   - 长期技术栈和职业信息
   - 长期使用偏好
   - 用户明确要求长期记住的信息
   - 已确认且具有持续价值的业务知识

3. 不需要记忆：
   - 普通问候
   - 一次性查询结果
   - 无持续价值的临时信息
   - 不确定的推测

4. 不要把所有用户输入都存入长期记忆。

5. 如果用户明确要求忘记或删除某条记忆，
   应进入记忆删除处理流程，而不是重新保存。

请返回符合结构化 Schema 的结果。
"""

DEFAULT_USER_ID = "default"


class MemoryItem(BaseModel):
    """One extracted memory decision."""

    action: Literal["save", "delete", "skip"] = Field(
        description=(
            "save=写入记忆; delete=用户要求忘记/删除; "
            "skip=不需要记忆或仅短期会话上下文"
        )
    )
    memory_type: Literal["short_term", "long_term", "none"] = Field(
        description="记忆类型；delete 时可标 long_term；无价值时用 none"
    )
    content: str = Field(
        description="要保存或删除的记忆内容（简洁、可独立理解的陈述句）"
    )
    reason: str = Field(default="", description="分类或操作理由")


class MemoryExtractionResult(BaseModel):
    items: list[MemoryItem] = Field(default_factory=list)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _message_to_text(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, BaseMessage):
        content = message.content
    elif isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = str(message)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content) if content is not None else ""


def _dialogue_snippet(
    messages: Sequence[Any],
    *,
    human_request: str | None = None,
    assistant_response: str | None = None,
    limit: int = 12,
) -> str:
    lines: list[str] = []
    if human_request:
        lines.append(f"用户: {human_request}")
    if assistant_response:
        lines.append(f"助手: {assistant_response}")

    for msg in list(messages)[-limit:]:
        if isinstance(msg, BaseMessage):
            role = {"human": "用户", "ai": "助手", "system": "系统", "tool": "工具"}.get(
                msg.type, msg.type
            )
            text = _message_to_text(msg)
        elif isinstance(msg, dict):
            raw = str(msg.get("type") or msg.get("role") or "unknown")
            role = {"human": "用户", "ai": "助手", "system": "系统", "tool": "工具"}.get(
                raw, raw
            )
            text = _message_to_text(msg)
        else:
            role, text = "未知", str(msg)
        if text.strip():
            lines.append(f"{role}: {text}")

    # de-dupe while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return "\n".join(unique) if unique else "(空对话)"


def get_memory_model() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def extract_memories(
    messages: Sequence[Any],
    *,
    human_request: str | None = None,
    assistant_response: str | None = None,
) -> MemoryExtractionResult:
    """Run the memory manager LLM and return structured decisions."""
    dialogue = _dialogue_snippet(
        messages,
        human_request=human_request,
        assistant_response=assistant_response,
    )
    llm = get_memory_model().with_structured_output(MemoryExtractionResult)
    result = llm.invoke(
        [
            SystemMessage(content=MEMORY_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "请分析以下对话，提取记忆决策（save/delete/skip）。\n\n"
                    f"{dialogue}"
                )
            ),
        ]
    )
    if isinstance(result, MemoryExtractionResult):
        return result
    return MemoryExtractionResult.model_validate(result)


def list_long_term_memories(
    user_id: str = DEFAULT_USER_ID,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    col = get_long_term_memory_collection()
    cursor = (
        col.find({"user_id": user_id})
        .sort("updated_at", -1)
        .limit(limit)
    )
    return list(cursor)


def format_long_term_context(
    user_id: str = DEFAULT_USER_ID,
    *,
    limit: int = 20,
) -> str:
    """Human-readable long-term memory block for system prompts."""
    docs = list_long_term_memories(user_id, limit=limit)
    if not docs:
        return ""
    lines = ["以下是该用户的长期记忆，请在回答时参考（不要逐条复述）："]
    for i, doc in enumerate(docs, start=1):
        content = str(doc.get("content") or "").strip()
        if content:
            lines.append(f"{i}. {content}")
    return "\n".join(lines) if len(lines) > 1 else ""


def save_long_term_memory(
    content: str,
    *,
    user_id: str = DEFAULT_USER_ID,
    reason: str = "",
    source_thread_id: str | None = None,
) -> dict[str, Any]:
    """Upsert a long-term memory by (user_id, content)."""
    content = content.strip()
    if not content:
        return {}

    col = get_long_term_memory_collection()
    now = _utcnow()
    memory_id = str(uuid4())
    result = col.update_one(
        {"user_id": user_id, "content": content},
        {
            "$set": {
                "user_id": user_id,
                "content": content,
                "memory_type": "long_term",
                "reason": reason,
                "source_thread_id": source_thread_id,
                "updated_at": now,
            },
            "$setOnInsert": {
                "memory_id": memory_id,
                "created_at": now,
            },
        },
        upsert=True,
    )
    doc = col.find_one({"user_id": user_id, "content": content})
    return {
        "upserted": result.upserted_id is not None,
        "matched": result.matched_count,
        "doc": doc,
    }


def delete_long_term_memories(
    content: str,
    *,
    user_id: str = DEFAULT_USER_ID,
) -> int:
    """Delete memories whose content matches or contains the query text."""
    content = content.strip()
    if not content:
        return 0
    col = get_long_term_memory_collection()
    # Prefer exact match; also remove near-matches via case-insensitive contains.
    exact = col.delete_many({"user_id": user_id, "content": content})
    fuzzy = col.delete_many(
        {
            "user_id": user_id,
            "content": {"$regex": re.escape(content), "$options": "i"},
        }
    )
    return exact.deleted_count + fuzzy.deleted_count


def apply_memory_extraction(
    extraction: MemoryExtractionResult,
    *,
    user_id: str = DEFAULT_USER_ID,
    source_thread_id: str | None = None,
) -> dict[str, Any]:
    """
    Persist long-term saves/deletes to MongoDB.

    short_term / skip items are ignored here — session short-term memory
    is handled by the LangGraph MongoDB checkpointer.
    """
    saved: list[str] = []
    deleted: list[str] = []
    skipped = 0

    for item in extraction.items:
        if item.action == "delete":
            n = delete_long_term_memories(item.content, user_id=user_id)
            if n:
                deleted.append(item.content)
            else:
                skipped += 1
            continue

        if item.action == "save" and item.memory_type == "long_term":
            save_long_term_memory(
                item.content,
                user_id=user_id,
                reason=item.reason,
                source_thread_id=source_thread_id,
            )
            saved.append(item.content)
            continue

        skipped += 1

    return {
        "saved": saved,
        "deleted": deleted,
        "skipped": skipped,
    }


def process_conversation_memories(
    messages: Sequence[Any],
    *,
    user_id: str = DEFAULT_USER_ID,
    source_thread_id: str | None = None,
    human_request: str | None = None,
    assistant_response: str | None = None,
) -> dict[str, Any]:
    """Extract memories from a turn and write long-term items to MongoDB."""
    extraction = extract_memories(
        messages,
        human_request=human_request,
        assistant_response=assistant_response,
    )
    applied = apply_memory_extraction(
        extraction,
        user_id=user_id,
        source_thread_id=source_thread_id,
    )
    return {
        "extraction": extraction.model_dump(),
        **applied,
    }


def resolve_user_id(
    config: Optional[dict[str, Any]] = None,
    explicit: str | None = None,
) -> str:
    if explicit:
        return explicit
    if config:
        configurable = config.get("configurable") or {}
        user_id = configurable.get("user_id")
        if user_id:
            return str(user_id)
    return DEFAULT_USER_ID


__all__ = [
    "MEMORY_SYSTEM_PROMPT",
    "DEFAULT_USER_ID",
    "MemoryItem",
    "MemoryExtractionResult",
    "extract_memories",
    "list_long_term_memories",
    "format_long_term_context",
    "save_long_term_memory",
    "delete_long_term_memories",
    "apply_memory_extraction",
    "process_conversation_memories",
    "resolve_user_id",
]
