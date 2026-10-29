"""
LangGraph agent graph.

- LLM is injected via provider factory (swap in .env, no code change).
- PostgreSQL checkpointer saves full conversation history per session.
  Set POSTGRES_ENABLED=true + POSTGRES_URL in .env to activate.
  Falls back to in-memory checkpointer when Postgres is unavailable.
- agent.log captures every node transition at DEBUG level.
"""

from __future__ import annotations
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.tools import ALL_TOOLS
from config.settings import settings
from core.provider_factory import get_llm_with_fallback

logger = logging.getLogger(__name__)


# ── Checkpointer ──────────────────────────────────────────────────────────────

def _build_checkpointer():
    if settings.postgres_enabled:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            checkpointer = PostgresSaver.from_conn_string(settings.postgres_url)
            checkpointer.setup()   # creates tables if they don't exist
            logger.info("LangGraph checkpointer: PostgreSQL (%s)", settings.postgres_url.split("@")[-1])
            return checkpointer
        except Exception:
            logger.exception(
                "PostgreSQL checkpointer failed — falling back to in-memory. "
                "Check POSTGRES_URL in .env."
            )

    logger.info("LangGraph checkpointer: in-memory (no persistence)")
    return MemorySaver()


# ── LLM (built once, shared across all requests) ──────────────────────────────

_llm_model      = get_llm_with_fallback(settings)
_llm_with_tools = _llm_model.bind_tools(ALL_TOOLS)

logger.info("LLM model ready with %d tools bound", len(ALL_TOOLS))


# ── Nodes ─────────────────────────────────────────────────────────────────────

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode

_tool_node = ToolNode(ALL_TOOLS)

MAX_TOOL_CALLS = settings.max_tool_calls


def call_llm(state: AgentState) -> dict:
    logger.debug("call_llm | question=%s | tool_call_count=%d",
                 state["question"], state.get("tool_call_count", 0))

    dist_cm   = state.get("face_status", {}).get("distance_cm")
    dist_info = f"{dist_cm} cm" if dist_cm is not None else "unknown"

    vision_event = state.get("vision_event", {})
    vision_info   = (
        f"Detected activity: {vision_event.get('gesture')} "
        f"(confidence: {int(vision_event.get('confidence', 0) * 100)}%, "
        f"source: {vision_event.get('source')})"
        if vision_event else "No activity detected"
    )

    system = SystemMessage(content=(
        "You are a fun, chill AI buddy connected to a face expression camera and a physical robot. "
        "Talk like a close friend — casual, witty, sometimes funny. No corporate tone ever.\n\n"
        "Current face status:\n"
        f"{state['face_status']}\n\n"
        f"Object distance (VL53L0X sensor): {dist_info}\n"
        f"Vision activity: {vision_info}\n"
        "If something came very close (under 30cm), react in a funny, surprised or sassy way.\n"
        "If a vision activity is detected, acknowledge it naturally and use tools if appropriate.\n\n"
        "Tools available:\n"
        "- shake_hand: handshake / haath milao\n"
        "- wave: wave / haath hilao\n"
        "- blink_led(times): blink LED\n"
        "- led_on / led_off: LED control\n"
        "- set_servo(angle): move servo 0-180°\n"
        "- get_weather(city): weather info\n"
        "- search_youtube(query): open YouTube\n"
        "- motor_move(direction, speed): move robot car (forward/backward/left/right/stop)\n\n"
        "Rules:\n"
        "1. Match the user's language exactly.\n"
        "2. Keep replies short and conversational.\n"
        "3. If a tool matches the request, call it once then confirm casually.\n"
        "4. Never sound like a customer support bot."
    ))

    human    = HumanMessage(content=state["question"])
    history  = state.get("messages", [])
    messages = [system, human] + history

    response   = _llm_with_tools.invoke(messages)
    tool_calls = len(response.tool_calls) if response.tool_calls else 0

    logger.debug(
        "call_llm done | tool_calls=%d | content_len=%d",
        tool_calls, len(response.content or ""),
    )

    return {
        "messages":        [response],
        "response":        response.content or "",
        "tool_call_count": state.get("tool_call_count", 0) + tool_calls,
    }


def run_tools(state: AgentState) -> dict:
    logger.debug("run_tools | executing %d tool call(s)",
                 len(state["messages"][-1].tool_calls))

    result        = _tool_node.invoke(state)
    tool_messages = result.get("messages", [])
    tool_outputs  = " | ".join(
        m.content for m in tool_messages if hasattr(m, "content")
    )

    logger.debug("run_tools done | outputs: %s", tool_outputs)

    return {"messages": tool_messages, "response": tool_outputs}


def should_use_tools(state: AgentState) -> str:
    if state.get("tool_call_count", 0) >= MAX_TOOL_CALLS:
        logger.warning("Max tool calls (%d) reached — forcing END", MAX_TOOL_CALLS)
        return "end"

    messages = state.get("messages", [])
    if not messages:
        return "end"

    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        logger.debug("Routing -> tools")
        return "tools"

    logger.debug("Routing -> end")
    return "end"


# ── Graph ─────────────────────────────────────────────────────────────────────

def create_agent():
    logger.info("Building LangGraph agent...")

    graph = StateGraph(AgentState)
    graph.add_node("llm",   call_llm)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", should_use_tools, {"tools": "tools", "end": END})
    graph.add_edge("tools", "llm")

    checkpointer = _build_checkpointer()
    compiled     = graph.compile(checkpointer=checkpointer)

    logger.info("LangGraph agent ready")
    return compiled


agent = create_agent()
