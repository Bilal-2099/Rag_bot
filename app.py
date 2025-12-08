import streamlit as st
import Rag_helpers as rh


# ---------------------------
# Load and index notes once
# ---------------------------
st.title("📚 RAG Study Assistant (with Quiz Mode)")

@st.cache_resource
def load_data():
    docs = rh.load_notes()
    if docs:
        docs = rh.build_index(docs)
    return docs

docs = load_data()

if not docs:
    st.warning("No notes found in 'My_notes' folder. Add .txt files and refresh.")
    st.stop()


# ---------------------------
# Sidebar Mode Selector
# ---------------------------
mode = st.sidebar.radio("Choose Mode", ["Ask AI", "Quiz Me"])

# -----------------------------------------
# MODE 1 — Ask AI (Normal RAG Chat)
# -----------------------------------------
if mode == "Ask AI":
    st.header("💬 Ask Your Notes Anything")

    # Memory across reruns
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display existing chat
    for turn in st.session_state.chat_history:
        st.chat_message("user").write(turn["user"])
        st.chat_message("assistant").write(turn["ai"])

    # User input
    user_query = st.chat_input("Ask something...")
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        context = rh.search_context(user_query, docs)

        with st.spinner("Thinking..."):
            answer = rh.ask_gemini(user_query, context, st.session_state.chat_history)

        with st.chat_message("assistant"):
            st.write(answer)

        # Save history
        st.session_state.chat_history.append({"user": user_query, "ai": answer})

        # Limit history to last 5
        if len(st.session_state.chat_history) > 5:
            st.session_state.chat_history.pop(0)



# -----------------------------------------
# MODE 2 — Quiz Mode
# -----------------------------------------
if mode == "Quiz Me":
    st.header("📝 Quiz Mode")

    # Initialize session state
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "current_context" not in st.session_state:
        st.session_state.current_context = None
    if "quiz_result" not in st.session_state:
        st.session_state.quiz_result = None

    # Button to start new question
    if st.button("🎯 New Question"):
        import random
        st.session_state.current_context = [random.choice(docs)]
        st.session_state.current_question = rh.generate_question(
            st.session_state.current_context
        )
        st.session_state.quiz_result = None

    # Show the question
    if st.session_state.current_question:
        st.subheader("Question:")
        st.info(st.session_state.current_question)

        user_answer = st.text_area("Your Answer:")

        if st.button("Check Answer"):
            with st.spinner("Checking..."):
                result = rh.check_answer(
                    user_answer,
                    st.session_state.current_context,
                    st.session_state.current_question,
                )

            st.session_state.quiz_result = result

    # Show result
    if st.session_state.quiz_result:
        result_text = st.session_state.quiz_result.strip().lower()

        if "incorrect" in result_text:
            st.error(st.session_state.quiz_result) # red
        else:
            st.success(st.session_state.quiz_result)   # green     


