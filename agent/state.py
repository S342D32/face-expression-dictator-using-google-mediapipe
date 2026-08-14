from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    question:        str
    face_status:     dict
    response:        str
    tool_call_count: int
    messages:        Annotated[list, add_messages]
