from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

# using sqllite checkpoint memory
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


# selecting our model
load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
)

# creating state graph to store our chat history throughout the state

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

# creating chat_node function

def chat_node(state:ChatState):
    messages= state['messages']
    response= model.invoke(messages)

    return {"messages": [response]}

# creating our graph and adding nodes and edges

graph=StateGraph(ChatState)
graph.add_node("chat_node",chat_node)

# adding edges
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node", END)

# adding checkpointer to save memory throughout the conversation
# creating sqllite checkpointer to save our conversation thread in local created db
conn=sqlite3.connect(database= "marvel_chatbot.db", check_same_thread =False)
checkpointer= SqliteSaver(conn=conn)



# compiling the graph 
chatbot= graph.compile(checkpointer=checkpointer)


# small function, from which we get the title for our chat thread
def generate_chat_title(user_message: str) -> str:
    """Create a short sidebar title from a chat's first user message."""

    prompt = f"""Create a concise title for this chat question.

Question: {user_message}

Rules:
- Return only the title.
- Use at most 10 words.
- Do not use quotation marks.
"""
    response = model.invoke([HumanMessage(content=prompt)])

    content=response.content

    if isinstance(content, list):
        title=" ".join(
            item.get("text","")
            for item in content
            if isinstance(item, dict) and item.get("type")=="text"
            
        )
    else:
        title=str(content)

    return " ".join(title.split())[:80] or "New Conversation"
