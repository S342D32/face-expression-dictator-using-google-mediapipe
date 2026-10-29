from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    question:        str
    face_status:     dict
    # Vision activity from phone ActivityTracker
    # Scale: add more fields here as new detectors are added
    vision_event:    dict   # { gesture, confidence, source, meta } or {}
    response:        str
    tool_call_count: int
    session_id:      str
    messages:        Annotated[list, add_messages]
