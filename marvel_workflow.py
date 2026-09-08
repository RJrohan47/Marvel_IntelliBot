"""Marvel's explicit LangGraph workflow.

The entry router isolates the general assistant from grounded YouTube RAG.
"""

from __future__ import annotations

from typing import Annotated, Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from marvel_tools import GENERAL_TOOLS
from youtube_rag.service import answer_video_question


load_dotenv()
MODEL_NAME = "gemini-3.6-flash"
model = ChatGoogleGenerativeAI(model=MODEL_NAME)

# tool binding 
general_model = model.bind_tools(GENERAL_TOOLS)


class ChatState(TypedDict):
    """Persistent state stored by the existing SQLite checkpointer."""

    messages: Annotated[list[BaseMessage], add_messages]
    active_video_id: NotRequired[str | None]


def route_initial_request(state: ChatState) -> Literal["general_agent", "youtube_rag"]:
    """Select the capability explicitly chosen by the thread's active context."""
    return "youtube_rag" if state.get("active_video_id") else "general_agent"


def general_agent_node(state: ChatState) -> dict:
    """General Gemini chat; it can decide whether to call GENERAL_TOOLS."""
    return {"messages": [general_model.invoke(state["messages"])]}


def youtube_rag_node(state: ChatState) -> dict:
    """Grounded video answer. General tools are never exposed on this route."""
    question = next((message.content for message in reversed(state["messages"]) if isinstance(message, HumanMessage)), "")
    if not question:
        return {"messages": [AIMessage(content="Please ask a question about the attached video.")]}
    try:
        answer = answer_video_question(str(question), state["active_video_id"])
    except Exception as exc:
        answer = f"I could not search the attached video transcript: {exc}"
    return {"messages": [AIMessage(content=answer)]}


def build_workflow(checkpointer):
    """Build the graph; dependency injection makes it straightforward to test."""
    graph = StateGraph(ChatState)

    # starting our flow 

    #creating nodes of our graph 
    graph.add_node("general_agent", general_agent_node)
    graph.add_node("general_tools", ToolNode(GENERAL_TOOLS))
    graph.add_node("youtube_rag", youtube_rag_node)


    # creating edges of our graph 
    graph.add_conditional_edges(START, route_initial_request, {"general_agent": "general_agent",
                                                                "youtube_rag": "youtube_rag"})
    
    graph.add_conditional_edges("general_agent", tools_condition, {"tools": "general_tools",
                                                                    END: END})
    
    graph.add_edge("general_tools", "general_agent")
    graph.add_edge("youtube_rag", END)

    return graph.compile(checkpointer=checkpointer)


def generate_chat_title(user_message: str) -> str:
    """Create a concise sidebar title without changing conversation state."""
    response = model.invoke([HumanMessage(content=("Create a concise title for this chat question. Return only the title, with at most 10 words.\n\nQuestion: " + user_message))])
    content = response.content
    if isinstance(content, list):
        content = " ".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")
    return " ".join(str(content).split())[:80] or "New Conversation"
