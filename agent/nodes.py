import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode

from .state import AgentState
from .tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# LLM + ToolNode — created once at module level
# ----------------------------------------------------------------

llm = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0.7,
)

llm_with_tools = llm.bind_tools(ALL_TOOLS)

tool_node = ToolNode(ALL_TOOLS)

MAX_TOOL_CALLS = 3   # hard stop — prevents infinite tool loops


# ----------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------

def call_llm(state: AgentState) -> dict:

    logger.info("call_llm node | question=%s", state["question"])

    system = SystemMessage(content=(
        "You are an AI assistant connected to a camera-based face "
        "expression detection system and a physical robot.\n\n"
        "Current face status:\n"
        f"{state['face_status']}\n\n"
        "Use the face status only when relevant. "
        "You have tools to control the robot — use them ONCE when the user asks "
        "for a physical action like shaking hands, waving, or blinking LED. "
        "After calling a tool, always give a short spoken response. "
        "Respond naturally and briefly."
    ))

    human = HumanMessage(content=state["question"])

    # Build message history — system + human always at front
    history  = state.get("messages", [])
    messages = [system, human] + history

    logger.info("Invoking LLM with tools...")

    response = llm_with_tools.invoke(messages)

    tool_calls = len(response.tool_calls) if response.tool_calls else 0

    logger.info(
        "LLM response | tool_calls=%d | content_len=%d",
        tool_calls,
        len(response.content or "")
    )

    return {
        "messages":        [response],
        "response":        response.content or "",
        "tool_call_count": state.get("tool_call_count", 0) + tool_calls,
    }


def run_tools(state: AgentState) -> dict:

    logger.info("run_tools node — executing tool calls")

    result = tool_node.invoke(state)

    tool_messages = result.get("messages", [])
    tool_outputs  = " | ".join(
        m.content for m in tool_messages if hasattr(m, "content")
    )

    logger.info("Tool outputs: %s", tool_outputs)

    return {
        "messages": tool_messages,
        "response": tool_outputs,
    }


# ----------------------------------------------------------------
# Routing
# ----------------------------------------------------------------

def should_use_tools(state: AgentState) -> str:

    # Hard stop — prevent infinite loops
    if state.get("tool_call_count", 0) >= MAX_TOOL_CALLS:
        logger.warning("Max tool calls reached — forcing END")
        return "end"

    messages = state.get("messages", [])

    if not messages:
        return "end"

    last = messages[-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        logger.info("Routing → tools")
        return "tools"

    logger.info("Routing → end")
    return "end"
