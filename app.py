import streamlit as st
from deploy import graph, save_message, load_history

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("AI Research Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []


# display previous UI messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])


user_question = st.chat_input("Ask something")


if user_question:

    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.write(user_question)

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

    with st.chat_message("assistant"):

        st.write(answer)

        if result.get("documents"):

            with st.expander("Sources"):

                for i, d in enumerate(result["documents"], 1):
                    st.write(f"Source {i}")
                    st.write(d[:500])