import streamlit as st
from langgraph_tool_backend import (
    chatbot,
    cleanup_expired_threads,
    delete_thread_memory,
    generate_chat_title,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from thread_manager import (
    fetch_thread_list,
    initialize_thread_table,
    delete_thread_record,
    save_thread_title,
    touch_thread,
)
from youtube_rag.service import clear_thread_video, get_thread_video, set_thread_video
import uuid


#--------------------------------Helper Functions ---------------------------------

# function to create dynamic thread_id
def generate_thread_id():
    # UUID4 strings are 36 characters, well below PostgresSaver's 255-character limit.
    return str(uuid.uuid4())

# function to create new chat session and clear message history

def new_chat_session():
    st.session_state['thread_id'] = generate_thread_id()
    st.session_state['message_history'] = []
    st.session_state['active_video_id'] = None
    st.session_state['active_video_url'] = ""

# creating a function where we will fetch the message history for particular thread_id and 
# it is achieveable by langgraph chatbot.get_state method

def get_message_history(thread_id):
    state=chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])


def message_text(content) -> str:
    """Return plain text from either a normal message or Gemini content blocks."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )

    return str(content or "")


def tool_display_name(tool_name: str) -> str:
    """Use readable labels for built-in tools and retain MCP tool names."""
    known_names = {
        "duckduckgo_search": "Web Search",
        "search": "Web Search",
        "search_tool": "Web Search",
        "calculator": "Calculator",
        "get_weather_info": "Weather",
    }
    return known_names.get(tool_name.casefold(), tool_name.replace("_", " ").replace("-", " ").title())


@st.dialog("Delete chat?")
def confirm_delete_chat() -> None:
    """Confirm and remove one chat, including its saved LangGraph state."""

    pending = st.session_state.get("chat_to_delete")
    if not pending:
        return

    st.write(f"Delete **{pending['title']}** and its saved thread history?")
    cancel_col, delete_col = st.columns(2)
    if cancel_col.button("Cancel", use_container_width=True):
        st.session_state["chat_to_delete"] = None
        st.rerun()

    if delete_col.button("Delete chat", type="primary", use_container_width=True):
        thread_id = pending["thread_id"]
        try:
            delete_thread_memory(thread_id)
            clear_thread_video(thread_id)
            delete_thread_record(thread_id)
        except Exception as exc:
            st.error(f"Could not completely delete this chat: {exc}")
            return

        st.session_state["thread_list"] = fetch_thread_list()
        if st.session_state["thread_id"] == thread_id:
            new_chat_session()
        st.session_state["chat_to_delete"] = None
        st.rerun()


# ------------------Creating MARVEL Chatbot UI using streamlit------------------------------


# first creating session_state using streamlit to store our message history

initialize_thread_table()
expired_thread_ids = cleanup_expired_threads()


#--------------------------- Session State for Message History---------------------------------
if 'message_history' not in st.session_state:
    st.session_state['message_history'] =[]  # it is a list which store dictionary of conversations

# creating session state for new thread_id

if 'thread_id' not in st.session_state:
    st.session_state['thread_id']= generate_thread_id() # unique thread id for each session to store

if st.session_state['thread_id'] in expired_thread_ids:
    new_chat_session()

# creating thread_list for session state to store all the threads created in a session
# here since we are using sqlite db for our threads, then for every session, everytime it executes , we will store list of threads we have in our db currently 

if 'thread_list' not in st.session_state:
    st.session_state['thread_list']=fetch_thread_list()
elif expired_thread_ids:
    st.session_state['thread_list'] = fetch_thread_list()

if 'active_video_id' not in st.session_state:
    st.session_state['active_video_id'] = None

if 'active_video_url' not in st.session_state:
    st.session_state['active_video_url'] = ""


#----------------------------Chatbot Sidebar User Interface --------------------------------

# create sidebar
st.sidebar.title("Marvel Chatbot")

# once it is clicked we get a new chat session 
if st.sidebar.button("New Chat"):
    new_chat_session()

st.sidebar.divider()
st.sidebar.subheader("YouTube video context")
video_url = st.sidebar.text_input(
    "YouTube URL or video ID",
    value=st.session_state['active_video_url'],
    placeholder="https://www.youtube.com/watch?v=...",
)
attach_video = st.sidebar.button("Attach and index video", use_container_width=True)
remove_video = st.sidebar.button(
    "Remove video context",
    disabled=not st.session_state['active_video_id'],
    use_container_width=True,
)

if attach_video:
    try:
        with st.spinner("Fetching captions and building the video index..."):
            video_id = set_thread_video(st.session_state['thread_id'], video_url)
        touch_thread(st.session_state['thread_id'])
        st.session_state['active_video_id'] = video_id
        st.session_state['active_video_url'] = f"https://www.youtube.com/watch?v={video_id}"
        st.success("Video context is ready. Your next questions will use its transcript.")
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))

if remove_video:
    clear_thread_video(st.session_state['thread_id'])
    touch_thread(st.session_state['thread_id'])
    st.session_state['active_video_id'] = None
    st.session_state['active_video_url'] = ""
    st.success("Video context removed. Questions will use the general assistant.")

if st.session_state['active_video_id']:
    st.sidebar.caption(f"Active video: {st.session_state['active_video_id']}")
    st.sidebar.info("Video context is active. Remove it to use general chat, web search, weather, or calculator tools.")

st.sidebar.header("Recent Chats")

# this for loop wil generate all the threads and also help user to switch between threads by creating actions

for thread in st.session_state['thread_list']:
    chat_col, delete_col = st.sidebar.columns([0.84, 0.16], vertical_alignment="center")
    if chat_col.button(thread['title'], key=thread['thread_id'], use_container_width=True):
        st.session_state['thread_id'] = thread['thread_id']
        touch_thread(thread['thread_id'])
        active_video = get_thread_video(thread['thread_id'])
        st.session_state['active_video_id'] = active_video['video_id'] if active_video else None
        st.session_state['active_video_url'] = active_video['canonical_url'] if active_video else ""
        message_history = get_message_history(thread['thread_id'])

        # the generated message history from our chatbot, is in dictionary but in different format then our message history (role,content) , so we will write logic to make it compatible
           
        temp_message_history=[]
        for msg in message_history:

            if isinstance(msg,HumanMessage):
                role='user'
            else:
                role='assistant'

            content = message_text(msg.content)

            temp_message_history.append({'role': role, 'content': content})

        st.session_state['message_history']=temp_message_history  

    if delete_col.button("🗑️", key=f"delete_{thread['thread_id']}", help="Delete this chat"):
        st.session_state["chat_to_delete"] = thread
        st.rerun()

if st.session_state.get("chat_to_delete"):
    confirm_delete_chat()



#----------------------------Main Chatbot User Interface------------------------------------


# loading our conversation history in ui, before the current conversation

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# now taking user input  using strealit chat input and now using streamlit chat_message we will show result

user_input=st.chat_input('Ask Anything')

if user_input:

    is_new_thread = not any(
        thread['thread_id'] == st.session_state['thread_id']
        for thread in st.session_state['thread_list']
    )

    # first add the message to message history

    st.session_state['message_history'].append({'role':'user', 'content':user_input})

    with st.chat_message('user'):
        st.markdown(user_input)

    # now getting response from our chatbot and adding streaming feature using langgraph stream method (like a generator) and using streamlit write_stream
    


    # creating config thread for current session chat
    CONFIG= {'configurable':{'thread_id':st.session_state['thread_id']}}

    CONFIG={
        "configurable":{"thread_id": st.session_state['thread_id']},
        "metadata": {"thread_id": st.session_state['thread_id']},
        "run_name": "chat_turn"
    }
    with st.chat_message('assistant'):
        progress = st.status("Thinking through your question...", expanded=False)
        active_stage = {"name": "thinking"}
        requested_tools = set()

        def report_tool_requested(tool_name: str) -> None:
            if not tool_name:
                return
            display_name = tool_display_name(tool_name)
            if tool_name not in requested_tools:
                requested_tools.add(tool_name)
                progress.write(f"Using **{display_name}**")
            progress.update(label="Using tools...", state="running", expanded=True)
            active_stage["name"] = "tools"

        def ai_text_stream():
            for message_chunk, metadata in chatbot.stream(
                {
                    'messages': [HumanMessage(content=user_input)],
                    'active_video_id': st.session_state['active_video_id'],
                },
                config=CONFIG,
                stream_mode='messages'
            ):
                # The graph also streams internal LTM extraction model chunks.
                # They are structured data for storage, not chat replies.
                if "ltm_internal" in (metadata.get("tags") or ()):
                    continue

                node = metadata.get("langgraph_node")
                if node == "general_agent":
                    # Tool names arrive in streamed tool-call chunks before ToolNode runs.
                    for tool_chunk in getattr(message_chunk, "tool_call_chunks", []) or []:
                        report_tool_requested(tool_chunk.get("name", ""))
                    # Some integrations provide complete calls rather than streamed chunks.
                    for tool_call in getattr(message_chunk, "tool_calls", []) or []:
                        report_tool_requested(tool_call.get("name", ""))
                elif node == "general_tools":
                    if isinstance(message_chunk, ToolMessage):
                        progress.update(label="Processing tool results...", state="running", expanded=True)
                        active_stage["name"] = "tools"
                    elif active_stage["name"] != "tools":
                        progress.update(label="Running requested tool...", state="running", expanded=True)
                        active_stage["name"] = "tools"
                elif node == "youtube_rag" and active_stage["name"] != "video":
                    progress.update(label="Searching the video transcript...", state="running")
                    active_stage["name"] = "video"

                if isinstance(message_chunk, AIMessage):
                    text = message_text(message_chunk.content)
                    if text:
                        if active_stage["name"] != "generating":
                            progress.update(label="Writing the response...", state="running")
                            active_stage["name"] = "generating"
                        yield text

        AI_response = st.write_stream(ai_text_stream())
        progress.update(label="Response ready", state="complete", expanded=False)

    # adding Ai response to message history
    st.session_state['message_history'].append({'role':'assistant', 'content':AI_response})

    if is_new_thread:
        title = generate_chat_title(user_input)
        save_thread_title(st.session_state['thread_id'], title)
    else:
        touch_thread(st.session_state['thread_id'])

    st.session_state['thread_list'] = fetch_thread_list()

   
