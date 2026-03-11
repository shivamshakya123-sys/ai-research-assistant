import streamlit as st
from deploy import graph, save_message, load_history

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Research Assistant")
st.write("Ask questions about your document database.")


if "messages" not in st.session_state:
    st.session_state.messages = []


# display UI history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


user_question = st.chat_input("Ask a question")


if user_question:

    # display user message
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.write(user_question)

    # load full DB history
    history = load_history()

    # include current question in history
    history_context = history + [f"User: {user_question}"]

    with st.spinner("Thinking..."):

        result = graph.invoke({
            "question": user_question,
            "chat_history": history_context
        })

    answer = result["answer"]

    # save messages
    save_message("User", user_question)
    save_message("Assistant", answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):

        st.write(answer)

        if "documents" in result and result["documents"]:

            with st.expander("Sources used"):

                for i, doc in enumerate(result["documents"], 1):

                    st.write(f"Source {i}:")
                    st.write(doc[:500])