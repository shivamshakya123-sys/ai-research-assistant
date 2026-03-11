from langchain_mistralai import ChatMistralAI
import os
from langchain_mistralai import MistralAIEmbeddings
import streamlit as st

import sqlite3

def init_memory():
    conn = sqlite3.connect("chat_memory.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        message TEXT
    )
    """)

    conn.commit()
    conn.close()

init_memory()




def save_message(role, message):
    conn = sqlite3.connect("chat_memory.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO conversations (role, message) VALUES (?, ?)",
        (role, message)
    )

    conn.commit()
    conn.close()


def load_history():
    conn = sqlite3.connect("chat_memory.db")
    cursor = conn.cursor()

    cursor.execute("SELECT role, message FROM conversations")

    rows = cursor.fetchall()
    conn.close()

    history = []
    for role, msg in rows:
        history.append(f"{role}: {msg}")

    return history





# creating llm
llm = ChatMistralAI(
    model="devstral-latest",
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
)







from langchain_mistralai import MistralAIEmbeddings

embed_model = MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
)






# retriving from the database DATA
from langchain_community.vectorstores import Chroma

@st.cache_resource
def load_vectorstore():
    return Chroma(
        persist_directory="./DATA",
        embedding_function=embed_model
    )

vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k":5})



# graphstate:
from typing import TypedDict, List

class GraphState(TypedDict):
    question: str
    chat_history: List[str]
    documents: List[str]
    answer: str
    route: str
    
    
    
    
# this will store previous messages:
def memory_node(state):

    history = state.get("chat_history", [])

    return {"chat_history": history}












# router node
def router_node(state):

    question = state["question"]

    prompt = f"""
Decide if the question requires searching a document database.

Return only:
select only one that requires:-
1.rag
2.llm

Question: {question}
"""

    response = llm.invoke(prompt)

    return {"route": response.content.strip().lower()}


# retrivel tool
def retrieval_node(state):

    question = state["question"]

    docs = retriever.invoke(question)

    documents = [doc.page_content for doc in docs]

    return {"documents": documents}






# answer generation
def answer_node(state):

    question = state["question"]
    context = "\n".join(state.get("documents", []))
    history = "\n".join(state.get("chat_history", []))

    prompt = f"""
    You are an AI research assistant.

    Conversation so far:
    {history}

    Context from documents:
    {context}

    User question:
    {question}

    Answer the user while considering the previous conversation.
    """

    response = llm.invoke(prompt)

    return {
    "answer": response.content,
    "documents": state.get("documents", [])
}





# llm
def llm_node(state):

    question = state["question"]

    response = llm.invoke(question)

    return {
        "answer": response.content,
        "documents": []
    }






def route_decision(state):

    if state["route"] == "rag":
        return "retrieval"
    else:
        return "llm"
    
    

# builing the graph
from langgraph.graph import StateGraph, END

builder = StateGraph(GraphState)

builder.add_node("memory", memory_node)
builder.add_node("router", router_node)
builder.add_node("retrieval", retrieval_node)
builder.add_node("answer", answer_node)
builder.add_node("llm", llm_node)

builder.set_entry_point("memory")

builder.add_edge("memory", "router")

builder.add_conditional_edges(
    "router",
    route_decision,
    {
        "retrieval": "retrieval",
        "llm": "llm"
    }
)
builder.add_edge("retrieval", "answer")
builder.add_edge("answer", END)
builder.add_edge("llm",END)




@st.cache_resource
def load_graph():
    return builder.compile()

graph = load_graph()
