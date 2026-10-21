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

from marvel_tools import EXPENSE_TOOLS, GENERAL_TOOLS
from model_fallback import build_chat_models, invoke_with_fallback
from short_term_memory import prepare_model_context
from youtube_rag.workflow import youtube_rag_graph


load_dotenv()
model, fallback_model = build_chat_models()

# Gemini is the primary model for all general chat and tool selection.
general_model = model.bind_tools(GENERAL_TOOLS)
fallback_general_model = fallback_model.bind_tools(GENERAL_TOOLS) if fallback_model else None

EXPENSE_TOOL_NAMES = {tool.name for tool in EXPENSE_TOOLS}

GENERAL_SYSTEM_PROMPT = """
You are a helpful, adaptable assistant.
Use the current user message and the conversation history for context. 
Choose an available tool when its description indicates it can provide needed information
or perform the requested action. Follow the tool's description and argument format;
when a tool requires a date, use YYYY-MM-DD. Do not invent tool results or claim an action
succeeded unless the tool output supports that. If a tool's required input is missing, or an
ambiguous detail could materially change the answer or action, ask one concise, targeted
follow-up before calling the tool. Do not guess dates, amounts, categories, or which record the
user means. For an update or deletion, use a record ID or uniquely identify the matching record;
if more than one record could match, ask the user which one. If a tool returns no matching data
or fails, say so clearly instead of filling the gap with a guess. Answer directly when the
available information is sufficient. For questions about the user's expense records, use the
expense tracker tools as the source of truth instead of relying on conversation history or summaries.

Current date: {current_date}. Use it when interpreting date-sensitive questions or relative dates.
"""

# only for the expense tools, not the general tools, because the general tools are not used in a way that requires this guidance

EXPENSE_RESULT_GUIDANCE = """
Use the expense tracker as the source of truth for expense records and amounts; do not use
conversation summaries as a substitute for querying it. The tracker uses Indian rupees (INR).
When reporting expense-tool results, use totals and category names exactly as returned. 
Do not describe one category as an alias, parent, or umbrella for another unless the tool output explicitly establishes
that relationship. Do not invent explanations for differences between category totals and the overall total; 
state that the returned values differ and show them. Do not claim the tool changed a record unless
its result confirms success.
"""


class ChatState(TypedDict):
    """Persistent state stored by the existing SQLite checkpointer."""

    messages: Annotated[list[BaseMessage], add_messages]
    active_video_id: NotRequired[str | None]
    model_route: NotRequired[Literal["gemini", "groq"]]
    context_summary: NotRequired[str]
    context_summary_message_count: NotRequired[int]


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


def _latest_turn_used_expense_tool(state: ChatState) -> bool:
    """Return whether this turn includes a call to one of the expense tools."""
    if not EXPENSE_TOOL_NAMES:
        return False

    messages = state["messages"]
    last_user_index = max(
        (index for index, message in enumerate(messages) if isinstance(message, HumanMessage)),
        default=-1,
    )
    for message in messages[last_user_index + 1:]:
        if isinstance(message, AIMessage):
            if any(
                call.get("name") in EXPENSE_TOOL_NAMES
                for call in (getattr(message, "tool_calls", None) or [])
            ):
                return True
        elif isinstance(message, ToolMessage) and message.name in EXPENSE_TOOL_NAMES:
            return True
    return False


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
    system_text = GENERAL_SYSTEM_PROMPT.format(current_date=date.today().isoformat())
    if _latest_turn_used_expense_tool(state):
        system_text += "\n\n" + EXPENSE_RESULT_GUIDANCE
    if any(
        isinstance(message, ToolMessage) and _tool_result_failed(message)
        for message in _latest_turn_tool_results(state)
    ):
        system_text += " A previous tool call failed. Do not repeat that same failed tool call; explain the limitation or use a different suitable tool."
    system = SystemMessage(content=system_text)
    history = _clean_gemini_history(state["messages"])
    context = prepare_model_context(
        system_message=system,
        history=history,
        summary=state.get("context_summary", ""),
        summarized_message_count=state.get("context_summary_message_count", 0),
        tools=GENERAL_TOOLS,
        summary_model=model,
        fallback_summary_model=fallback_model,
    )
    return {
        "model_route": "gemini",
        "messages": [invoke_with_fallback(general_model, fallback_general_model, context.messages)],
        "context_summary": context.summary,
        "context_summary_message_count": context.summarized_message_count,
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
