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
    documents: List[str]
    answer: str
    route: str


# ---------------- ROUTER ----------------

def router_node(state):

    question = state["question"].lower()

    retrieval_keywords = [
        "pdf",
        "document",
        "paper",
        "file",
        "database",
        "from my data",
        "from the docs",
        "search my data",
        "search database"
    ]

    if any(word in question for word in retrieval_keywords):
        route = "rag"
    else:
        route = "llm"

    return {"route": route}


# ---------------- RETRIEVAL ----------------

def retrieval_node(state):

    question = state["question"]

    docs = retriever.invoke(question)

    documents = [d.page_content for d in docs]

    return {"documents": documents}


# ---------------- ANSWER NODE (RAG) ----------------

def answer_node(state):

    question = state["question"]
    docs = state.get("documents", [])

    context = "\n\n".join(docs)

    history_rows = load_history()

    messages = [
        {
            "role": "system",
            "content":
            """
You are an AI assistant.

If context is provided, it comes from the user's uploaded files.
Use the context to answer the question.

Do not mention retrieval or sources.
Just answer naturally.
"""
        }
    ]

    for role, msg in history_rows:

        if role.lower() == "user":
            messages.append({"role": "user", "content": msg})
        else:
            messages.append({"role": "assistant", "content": msg})

    messages.append({
        "role": "user",
        "content": f"""
Context:
{context}

Question:
{question}
"""
    })

    response = llm.invoke(messages)

    return {
        "answer": response.content
    }


# ---------------- SIMPLE LLM ----------------

def llm_node(state):

    question = state["question"]

    history_rows = load_history()

    messages = [
        {
            "role": "system",
            "content":
            """
You are a helpful AI assistant.

Use conversation history to answer naturally.
Only rely on general knowledge.
Do not access any document database unless explicitly requested.
"""
        }
    ]

    for role, msg in history_rows:

        if role.lower() == "user":
            messages.append({"role": "user", "content": msg})
        else:
            messages.append({"role": "assistant", "content": msg})

    messages.append({"role": "user", "content": question})

    response = llm.invoke(messages)

    return {
        "answer": response.content
    }


# ---------------- ROUTE DECISION ----------------

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