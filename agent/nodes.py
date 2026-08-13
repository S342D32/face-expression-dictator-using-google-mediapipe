
from dotenv import load_dotenv

load_dotenv()
from .state import AgentState

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
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