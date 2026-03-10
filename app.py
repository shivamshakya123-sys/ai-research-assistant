import streamlit as st
from deploy import graph

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Research Assistant")
st.write("Ask questions about your document database.")

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# input box
user_question = st.chat_input("Ask a question")

if user_question:

    # show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.write(user_question)

    # run graph
    with st.spinner("Thinking..."):
        result = graph.invoke({
            "question": user_question,
            "chat_history": st.session_state.chat_history
        })

    answer = result["answer"]

    # update memory
    st.session_state.chat_history.append(user_question)

    # store assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    # show assistant message
    with st.chat_message("assistant"):
        st.write(answer)

        if "documents" in result and result["documents"]:
            with st.expander("Sources used"):
                for i, doc in enumerate(result["documents"], 1):
                    st.write(f"Source {i}:")
                    st.write(doc[:500])