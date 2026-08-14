import logging
from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import call_llm, run_tools, should_use_tools

logger = logging.getLogger(__name__)


def create_agent():

    logger.info("Building LangGraph agent...")

    graph = StateGraph(AgentState)

    graph.add_node("llm",   call_llm)
    graph.add_node("tools", run_tools)

    graph.add_edge(START, "llm")

    graph.add_conditional_edges(
        "llm",
        should_use_tools,
        {
            "tools": "tools",
            "end":   END,
        }
    )

    # After tools run, go back to LLM so it can form a final response
    graph.add_edge("tools", "llm")

    compiled = graph.compile()

    logger.info("LangGraph agent ready")

    return compiled


agent = create_agent()
