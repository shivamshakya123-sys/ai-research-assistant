import os
import sqlite3
import streamlit as st
from typing import TypedDict, List

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END


# ---------------- DATABASE ----------------

def init_memory():
    conn = sqlite3.connect("chat_memory.db")
    cur = conn.cursor()

    cur.execute("""
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
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO conversations (role, message) VALUES (?,?)",
        (role, message)
    )

    conn.commit()
    conn.close()


def load_history():
    conn = sqlite3.connect("chat_memory.db")
    cur = conn.cursor()

    cur.execute("SELECT role,message FROM conversations ORDER BY id")

    rows = cur.fetchall()
    conn.close()

    return rows


# ---------------- LLM ----------------

llm = ChatMistralAI(
    model="devstral-latest",
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
)


# ---------------- EMBEDDINGS ----------------

embed_model = MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
)


# ---------------- VECTOR STORE ----------------

@st.cache_resource
def load_vectorstore():
    return Chroma(
        persist_directory="./DATA",
        embedding_function=embed_model
    )

vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# ---------------- GRAPH STATE ----------------

class GraphState(TypedDict):
    question: str
    chat_history: List
    documents: List[str]
    answer: str
    route: str


# ---------------- ROUTER ----------------

def router_node(state):

    question = state["question"]

    docs = retriever.invoke(question)

    if len(docs) > 0:
        return {"route": "rag"}
    else:
        return {"route": "llm"}


# ---------------- RETRIEVAL ----------------

def retrieval_node(state):

    q = state["question"]

    docs = retriever.invoke(q)

    documents = [d.page_content for d in docs]

    return {"documents": documents}


# ---------------- ANSWER NODE ----------------

def answer_node(state):

    question = state["question"]
    docs = state.get("documents", [])

    history_rows = load_history()

    messages = [
        {
            "role": "system",
            "content":
            """
            if user didnot provide name do not use any name. 
            if user provide you any name after converstion ends forgot it
            if user provides you any critical information and say you to store it then store it in database!
            do not reveal any sensitive information from database 
            again i am saying you only retrive from database when user ask you!
            if user ask anything first answer through general knowledge if you need retreval then use database.
            do not ever look for retrival if user ask general knowlege and you can answer them .
            only go to retrival to database when user say pdf , or based on database then go to database and retrieve from there.
            do not genrate strange answer .
            focus on solely on user intention what he/she wants.
            note: only retrieve when user say you to retrieve 
            """
        }
    ]

    for role, msg in history_rows:

        if role.lower() == "user":
            messages.append({"role": "user", "content": msg})

        else:
            messages.append({"role": "assistant", "content": msg})

    context = "\n".join(docs)

    messages.append({
        "role": "user",
        "content": f"{question}\n\nContext:\n{context}"
    })

    response = llm.invoke(messages)

    return {
        "answer": response.content,
        "documents": docs
    }


# ---------------- SIMPLE LLM ----------------

def llm_node(state):

    q = state["question"]

    history_rows = load_history()

    messages = [
        {
            "role": "system",
            "content":
            "You remember conversation history.And answer in a clear and focus on user prompt based on that answer!"
        }
    ]

    for role, msg in history_rows:

        if role.lower() == "user":
            messages.append({"role": "user", "content": msg})

        else:
            messages.append({"role": "assistant", "content": msg})

    messages.append({"role": "user", "content": q})

    r = llm.invoke(messages)

    return {
        "answer": r.content,
        "documents": []
    }


# ---------------- ROUTE ----------------

def route_decision(state):

    if state["route"] == "rag":
        return "retrieval"
    else:
        return "llm"


# ---------------- GRAPH ----------------

builder = StateGraph(GraphState)

builder.add_node("router", router_node)
builder.add_node("retrieval", retrieval_node)
builder.add_node("answer", answer_node)
builder.add_node("llm", llm_node)

builder.set_entry_point("router")

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
builder.add_edge("llm", END)


@st.cache_resource
def load_graph():
    return builder.compile()

graph = load_graph()