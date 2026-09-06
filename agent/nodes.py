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

MAX_TOOL_CALLS = 3


# ----------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------

def call_llm(state: AgentState) -> dict:

    logger.info("call_llm node | question=%s", state["question"])

    dist_cm   = state.get("face_status", {}).get("distance_cm")
    dist_info = f"{dist_cm} cm" if dist_cm is not None else "unknown"

    system = SystemMessage(content=(
        "You are a fun, chill AI buddy connected to a face expression camera and a physical robot. "
        "Talk like a close friend — casual, witty, sometimes funny. No corporate tone ever.\n\n"
        "Current face status:\n"
        f"{state['face_status']}\n\n"
        f"Object distance (VL53L0X sensor): {dist_info}\n"
        "If something came very close (under 30cm), react in a funny, surprised or sassy way. "
        "Like 'arre bhai itne paas kyu aa gaye!' or 'yo that's way too close man!' — be creative.\n\n"
        "Tools available:\n"
        "- shake_hand: handshake / haath milao\n"
        "- wave: wave / haath hilao\n"
        "- blink_led(times): blink LED / light blink karo\n"
        "- led_on: LED on / light jalao\n"
        "- led_off: LED off / light band karo\n"
        "- set_servo(angle): move servo / servo ghoomao\n"
        "- get_weather(city): weather / mausam\n"
        "- search_youtube(query): open YouTube with search\n\n"
        "Rules:\n"
        "1. Match the user's language exactly — Hindi reply for Hindi, Odia reply for Odia, English for English, mix if they mix.\n"
        "2. Keep replies short and conversational, like texting a friend.\n"
        "3. If a tool matches the request, call it once then confirm casually.\n"
        "4. Never sound like a customer support bot or a formal assistant."
    ))

    human = HumanMessage(content=state["question"])

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

    if state.get("tool_call_count", 0) >= MAX_TOOL_CALLS:
        logger.warning("Max tool calls reached — forcing END")
        return "end"

    messages = state.get("messages", [])

    if not messages:
        return "end"

    last = messages[-1]

    if hasattr(last, "tool_calls") and last.tool_calls:
        logger.info("Routing -> tools")
        return "tools"

    logger.info("Routing -> end")
    return "end"
