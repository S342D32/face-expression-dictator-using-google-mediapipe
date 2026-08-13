from typing import TypedDict


class AgentState(TypedDict):
    question: str
    face_status: dict
    response: str