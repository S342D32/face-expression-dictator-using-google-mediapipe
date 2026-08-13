from langchain_openai import ChatOpenAI

from .state import AgentState


llm = ChatOpenAI(
    model="gpt-5.4",
    temperature=0.7
)


def call_llm(state: AgentState):

    prompt = f"""
You are an AI assistant connected to a camera-based
face and expression detection system.

User question:
{state["question"]}

Current user status:
{state["face_status"]}

Use the user's visual status only when it is relevant.

Respond naturally and briefly.
"""

    response = llm.invoke(prompt)

    return {
        "response": response.content
    }