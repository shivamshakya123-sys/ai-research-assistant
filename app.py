import streamlit as st
from deploy import graph, save_message

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------- UI STYLE ----------------

st.markdown("""
<style>

body {
    background-color:#0e1117;
}

.chat-user {
    background:#2563eb;
    padding:12px;
    border-radius:12px;
    margin:8px 0;
    color:white;
}

.chat-assistant {
    background:#1f2937;
    padding:12px;
    border-radius:12px;
    margin:8px 0;
    color:white;
}

.source-box {
    background:#111827;
    border-left:4px solid #4f46e5;
    padding:10px;
    border-radius:8px;
    margin-bottom:10px;
}

.header-title{
    font-size:40px;
    font-weight:700;
}

.subtitle{
    color:#9ca3af;
    font-size:16px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown(
    '<div class="header-title">AI Research Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions about your knowledge base and previous conversations.</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("Controls")

    if st.button("Clear Chat UI"):
        st.session_state.messages = []

    st.markdown("---")

    st.markdown("### About")

    st.write(
        "This assistant combines retrieval search and conversation memory."
    )

    st.write(
        "Built using LangGraph and Mistral models."
    )

# ---------------- CHAT MEMORY ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- DISPLAY CHAT ----------------

for m in st.session_state.messages:

    if m["role"] == "user":
        st.markdown(
            f'<div class="chat-user">{m["content"]}</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f'<div class="chat-assistant">{m["content"]}</div>',
            unsafe_allow_html=True
        )

# ---------------- INPUT ----------------

user_question = st.chat_input("Ask a question")

# ---------------- MAIN LOGIC ----------------

if user_question:

    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    st.markdown(
        f'<div class="chat-user">{user_question}</div>',
        unsafe_allow_html=True
    )

    save_message("user", user_question)

    with st.spinner("Thinking..."):

        result = graph.invoke({
            "question": user_question
        })

    answer = result["answer"]

    save_message("assistant", answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    st.markdown(
        f'<div class="chat-assistant">{answer}</div>',
        unsafe_allow_html=True
    )

    # ---------------- SOURCES ----------------

    if result.get("documents"):

        with st.expander("Sources used"):

            for i, doc in enumerate(result["documents"],1):

                st.markdown(
                    f'<div class="source-box"><b>Source {i}</b><br>{doc[:500]}</div>',
                    unsafe_allow_html=True
                )