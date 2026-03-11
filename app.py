import streamlit as st
from deploy import graph, save_message, load_history

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Research Assistant")
st.write("Ask questions about your document database.")

# Load persistent history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_history()

# UI message history
if "messages" not in st.session_state:
    st.session_state.messages = []

    # rebuild UI messages from stored history
    for item in st.session_state.chat_history:
        if item.startswith("User:"):
            st.session_state.messages.append({
                "role": "user",
                "content": item.replace("User:", "").strip()
            })
        elif item.startswith("Assistant:"):
            st.session_state.messages.append({
                "role": "assistant",
                "content": item.replace("Assistant:", "").strip()
            })

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
user_question = st.chat_input("Ask a question")

if user_question:

    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.write(user_question)

    # Run LangGraph
    with st.spinner("Thinking..."):
        result = graph.invoke({
            "question": user_question,
            "chat_history": st.session_state.chat_history
        })

    answer = result["answer"]

    # Save to persistent database
    save_message("User", user_question)
    save_message("Assistant", answer)

    # Update session memory
    st.session_state.chat_history.append(f"User: {user_question}")
    st.session_state.chat_history.append(f"Assistant: {answer}")

    # Store assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    # Show assistant message
    with st.chat_message("assistant"):
        st.write(answer)

        if "documents" in result and result["documents"]:
            with st.expander("Sources used"):
                for i, doc in enumerate(result["documents"], 1):
                    st.write(f"Source {i}:")
                    st.write(doc[:500])