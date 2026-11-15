# Lesson 3: Async MCP Tools with HITL Approval
# Async API with MCP tool approval using ainvoke()

from fastapi import APIRouter
from uuid import uuid4
from app.models import StartRequest, GraphResponse, ApproveRequest
from app.mcp_agent import get_agent
from app.memory import (
    DEFAULT_USER_ID,
    process_conversation_memories,
    resolve_user_id,
)
from langgraph.types import Command
from langchain_core.messages import HumanMessage

router = APIRouter()


async def run_agent_and_response(input_data, config, *, user_id: str):
    """
    Run the agent and return a response based on the state.
    Handles both initial requests and resuming after approval.
    """
    agent = await get_agent(user_id=user_id)
    response = await agent.ainvoke(input_data, config)
    thread_id = config["configurable"]["thread_id"]

    # Check if interrupted for tool approval
    if "__interrupt__" in response:
        interrupts = response["__interrupt__"]
        interrupt_data = interrupts[0].value

        # Extract tool details
        tool_name = interrupt_data.get("awaiting", "unknown")
        tool_args = interrupt_data.get("args", {})
        assistant_response = (
            f"I want to call the tool '{tool_name}' with arguments: {tool_args}"
        )

        return GraphResponse(
            thread_id=thread_id,
            run_status="user_feedback",
            assistant_response=assistant_response,
        )

    # Graph finished - extract final response and update long-term memory
    assistant_response = response["messages"][-1].content
    try:
        process_conversation_memories(
            response.get("messages") or [],
            user_id=user_id,
            source_thread_id=thread_id,
            assistant_response=(
                assistant_response
                if isinstance(assistant_response, str)
                else str(assistant_response)
            ),
        )
    except Exception:
        pass

    return GraphResponse(
        thread_id=thread_id,
        run_status="finished",
        assistant_response=assistant_response,
    )


@router.post("/mcp/start", response_model=GraphResponse)
async def start_mcp_agent(request: StartRequest):
    """
    Start a new MCP agent conversation.
    """
    thread_id = str(uuid4())
    user_id = resolve_user_id(explicit=request.user_id) or DEFAULT_USER_ID
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    input_data = {"messages": [HumanMessage(content=request.human_request)]}

    return await run_agent_and_response(input_data, config, user_id=user_id)


@router.post("/mcp/approve", response_model=GraphResponse)
async def approve_mcp_tool(request: ApproveRequest):
    """
    Approve or reject a tool execution and continue the agent.
    """
    user_id = resolve_user_id(explicit=request.user_id) or DEFAULT_USER_ID
    config = {
        "configurable": {
            "thread_id": request.thread_id,
            "user_id": user_id,
        }
    }
    decision = {"approved": request.approve_action == "approved"}
    input_data = Command(resume=decision)

    return await run_agent_and_response(input_data, config, user_id=user_id)
