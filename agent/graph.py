from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import call_llm


def create_agent():

    graph = StateGraph(AgentState)

    graph.add_node("llm", call_llm)

    graph.add_edge(START, "llm")
    graph.add_edge("llm", END)

    return graph.compile()


agent = create_agent()