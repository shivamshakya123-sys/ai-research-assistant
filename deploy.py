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
            """You remember conversation history. Use it when answering. just use conversation history when you need answering the question like what is , what was , or what i asked use converstation history never answer only 'dont know' when you dont know the answer! And also dont make unsusual expression only focus on what user provides and dont make strange answer like when use say something like tranformers or any name of the subject first look at the retrieval and then
            if you not found that then you can generate the answer and never explain in long paragraph unless user say you to explain in paragraph always use brief explanations! 
            and first look and database search there for you answer if you dont find relevent chunks then generate based on your explanation. Note:-Dont use any name unless user provides you any name or he/she said thier name then answer do not take out any name at start of the conversation only use user provided name !"""
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