"""Marvel's explicit LangGraph workflow.

The entry router isolates the general assistant from grounded YouTube RAG.
"""

from __future__ import annotations

from typing import Annotated, Literal, NotRequired, TypedDict
from datetime import date

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from marvel_tools import GENERAL_TOOLS
from model_fallback import build_chat_models, invoke_with_fallback
from youtube_rag.workflow import youtube_rag_graph


load_dotenv()
model, fallback_model = build_chat_models()

# Gemini is the primary model for all general chat and tool selection.
general_model = model.bind_tools(GENERAL_TOOLS)
fallback_general_model = fallback_model.bind_tools(GENERAL_TOOLS) if fallback_model else None


class ChatState(TypedDict):
    """Persistent state stored by the existing SQLite checkpointer."""

    messages: Annotated[list[BaseMessage], add_messages]
    active_video_id: NotRequired[str | None]
    model_route: NotRequired[Literal["gemini", "groq"]]


def route_initial_request(state: ChatState) -> Literal["general_agent", "youtube_rag"]:
    """Select the capability explicitly chosen by the thread's active context."""
    return "youtube_rag" if state.get("active_video_id") else "general_agent"


def _tool_result_failed(message: ToolMessage) -> bool:
    if getattr(message, "status", None) == "error":
        return True
    content = message.content
    if isinstance(content, list):
        content = " ".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    if isinstance(content, dict):
        return bool(content.get("error"))
    text = str(content).strip().casefold()
    return text.startswith(("[tool_error]", "error:", "error invoking tool")) or '"error"' in text[:160]


def _latest_turn_tool_results(state: ChatState) -> list[ToolMessage]:
    messages = state["messages"]
    last_user_index = max(
        (index for index, message in enumerate(messages) if isinstance(message, HumanMessage)),
        default=-1,
    )
    return [
        message for message in messages[last_user_index + 1:]
        if isinstance(message, ToolMessage)
    ]


def _clean_gemini_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Drop failed/orphaned tool exchanges that can poison a later provider request."""
    tool_call_ids = {
        call.get("id")
        for message in messages
        if isinstance(message, AIMessage)
        for call in (getattr(message, "tool_calls", None) or [])
        if call.get("id")
    }
    result_ids = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage) and message.tool_call_id
    }
    failed_ids = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage) and message.tool_call_id and _tool_result_failed(message)
    }

    bad_call_ids = set(failed_ids)
    for message in messages:
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            ids = {call.get("id") for call in message.tool_calls if call.get("id")}
            if not ids or not ids.issubset(result_ids) or ids.intersection(failed_ids):
                bad_call_ids.update(ids)

    clean_messages: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            if not message.tool_call_id or message.tool_call_id not in tool_call_ids or message.tool_call_id in bad_call_ids:
                continue
        elif isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            ids = {call.get("id") for call in message.tool_calls if call.get("id")}
            if not ids or ids.intersection(bad_call_ids):
                continue
        clean_messages.append(message)
    return clean_messages


def general_agent_node(state: ChatState) -> dict:
    """General Gemini chat; it can decide whether to call GENERAL_TOOLS."""
    system_text = (
        f"Today's date is {date.today().isoformat()}. "
        "For expense questions, use the expense tools and pass dates as YYYY-MM-DD. "
        "Report expense totals and category breakdowns from the tool results. Treat each category "
        "exactly as returned by the tools; do not claim that one category is an alias, parent, "
        "or umbrella for another unless a tool explicitly confirms that relationship. Do not "
        "invent explanations for category differences. If the category totals do not add up to "
        "the reported overall total, say that the tool results differ and show the returned values."
    )
    if any(
        isinstance(message, ToolMessage) and _tool_result_failed(message)
        for message in _latest_turn_tool_results(state)
    ):
        system_text += " A previous tool call failed. Do not repeat that same failed tool call; explain the limitation or use a different suitable tool."
    system = SystemMessage(content=system_text)
    history = _clean_gemini_history(state["messages"])
    return {
        "model_route": "gemini",
        "messages": [invoke_with_fallback(general_model, fallback_general_model, [system, *history])],
    }


def youtube_rag_node(state: ChatState) -> dict:
    """Adapt Marvel state to the private-state YouTube RAG subgraph."""
    question = next((message.content for message in reversed(state["messages"]) if isinstance(message, HumanMessage)), "")
    if not question:
        return {"messages": [AIMessage(content="Please ask a question about the attached video.")]}
    video_id = state.get("active_video_id")

    if not video_id:
        return {"messages": [AIMessage(content="Please attach a YouTube video first.")]}
    
    try:
        # The child graph has its own state schema. This wrapper explicitly
        # maps only the question/video ID in, then maps the answer back to chat.
        result = youtube_rag_graph.invoke({
            "question": str(question),
            "video_id": video_id,
        })
        answer = result["answer"]
    except Exception as exc:
        answer = f"I could not search the attached video transcript: {exc}"
    return {"messages": [AIMessage(content=answer)]}


def build_workflow(checkpointer):
    """Build the graph; dependency injection makes it straightforward to test."""
    graph = StateGraph(ChatState)

    # starting our flow 

    #creating nodes of our graph 
    graph.add_node("general_agent", general_agent_node)
    graph.add_node("general_tools", ToolNode(GENERAL_TOOLS, handle_tool_errors=True))
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
    prompt = [HumanMessage(content=("Create a concise title for this chat question. Return only the title, with at most 10 words.\n\nQuestion: " + user_message))]
    response = invoke_with_fallback(model, fallback_model, prompt)
    content = response.content
    if isinstance(content, list):
        content = " ".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")
    return " ".join(str(content).split())[:80] or "New Conversation"

# print(build_workflow(checkpointer=None).get_graph(xray=True).draw_ascii())
