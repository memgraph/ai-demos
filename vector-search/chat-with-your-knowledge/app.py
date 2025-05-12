import streamlit as st
from controller import StorageController, LLMController

@st.cache_resource
def get_controller():
    return StorageController()


@st.cache_resource
def get_llm_controller():
    return LLMController()


controller = get_controller()
llm_controller = get_llm_controller()

# --- Sidebar Navigation ---
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Ingest Wikipedia",
        "Ingest by Yourself",
        "Dataset Exploration",
        "Chat With Your Knowledge",
        "Generate Pub Quiz",
    ],
)


# --- Shared helpers ---
def difficulty_flag(level: str) -> str:
    level = level.lower()
    return {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(level, "⚪️")


# --- Shared language prefix input ---
st.sidebar.markdown("### Language Settings")
lang_prefix = st.sidebar.text_input("Optional language prefix", value="en")

# ==============================
# 📥 Ingest Wikipedia
# ==============================
if page == "Ingest Wikipedia":
    st.title("📥 Ingest Wikipedia Page into Memgraph")

    with st.form("ingest_form"):
        category = st.text_input("Enter Wikipedia page title", value="")
        save_as_category = st.text_input(
            "Save as category (empty will save with same name)", value=""
        )
        ingestion_mode = st.radio(
            "Ingestion mode", options=["Ingest from scratch", "Update dataset"], index=0
        )
        section_filter = st.text_input(
            "Target section (e.g. Plot, Reception, Cast)", value=""
        )
        submitted = st.form_submit_button("Ingest")

        if submitted:
            with st.spinner("🔄 Ingesting and creating vector index..."):
                mode = (
                    "replace" if ingestion_mode == "Ingest from scratch" else "append"
                )
                has_section_filter = (
                    section_filter is not None and len(section_filter) > 0
                )
                method = "detailed" if has_section_filter else "quick"
                section = section_filter
                count = controller.ingest_wikipedia(
                    category,
                    save_as_category,
                    lang_prefix,
                    mode=mode,
                    method=method,
                    section_filter=section_filter if has_section_filter else None,
                )
                if count is not None:
                    verb = "Replaced" if mode == "replace" else "Appended"
                    st.success(
                        f"✅ {verb} {count} paragraphs from '{category}' into storage."
                    )
                else:
                    st.success(
                        f"✅ Paragraphs from '{category}' already exist in storage!"
                    )

# ==============================
# ✍️ Ingest by Yourself
# ==============================
elif page == "Ingest by Yourself":
    st.title("✍️ Ingest a Custom Paragraph")

    available_categories = controller.get_all_categories()

    with st.form("custom_ingest_form"):
        st.markdown("#### Paste your content")
        user_paragraph = st.text_area("Text to ingest", height=300)

        st.markdown("#### Choose where to save it")
        existing_label = st.selectbox(
            "Save to existing label:", options=available_categories + [""]
        )
        new_label = st.text_input(
            "Or enter a new label (will override above if filled):"
        )

        submitted = st.form_submit_button("📥 Ingest Text")

        if submitted:
            if not user_paragraph.strip():
                st.warning("⚠️ Please paste some text.")
            else:
                target_label = (
                    new_label.strip() if new_label.strip() else existing_label
                )
                if not target_label:
                    st.warning("⚠️ Please select or enter a category name.")
                else:
                    with st.spinner("Embedding and saving..."):
                        count = controller.ingest_custom_text(
                            target_label,
                            user_paragraph,
                            lang_prefix=lang_prefix,
                            mode="append",
                        )
                        st.success(f"✅ Ingested 1 paragraph into '{target_label}'.")

# ==============================
# 📊 Dataset Exploration
# ==============================
elif page == "Dataset Exploration":
    st.title("📊 Explore Your Ingested Dataset")

    available_categories = controller.get_all_categories()
    if not available_categories:
        st.info("ℹ️ No datasets found. Please ingest something first.")
    else:
        selected_category = st.selectbox(
            "Select a category to explore:", options=available_categories
        )
        if st.button("🔍 Retrieve Dataset"):
            with st.spinner(f"Retrieving paragraphs from '{selected_category}'..."):
                paragraphs = controller.get_all_paragraphs_from_category(
                    selected_category
                )
                if not paragraphs:
                    st.warning("No paragraphs found for the selected category.")
                else:
                    st.success(f"✅ Found {len(paragraphs)} paragraphs.")
                    for i, item in enumerate(paragraphs):
                        with st.expander(f"📄 Paragraph {i+1}", expanded=False):
                            st.markdown(item["content"])


# ==============================
# 💬 Chat With Your Knowledge (Chatbot)
# ==============================
elif page == "Chat With Your Knowledge":
    st.title("💬 Chat with Your Knowledge")

    available_categories = controller.get_all_categories()
    if not available_categories:
        st.info("ℹ️ No categories ingested yet. Please ingest some data first.")
    else:
        category = st.selectbox(
            "Select a page to chat with:", options=available_categories
        )

        # Initialize chat history in session
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display previous chat messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input box
        user_input = st.chat_input("Ask a question about the selected page...")
        if user_input:
            st.chat_message("user").markdown(user_input)
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            # Semantic search
            with st.spinner("🔍 Retrieving relevant knowledge..."):
                context = controller.get_similar_documents(category, user_input, 10)

            # Generate answer
            with st.spinner("🧠 GPT-4o is thinking..."):
                answer = llm_controller.answer_question_based_on_excerpts(
                    user_input, context, lang_prefix
                )

            # Display bot response
            st.chat_message("assistant").markdown(answer)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

            # Optional: display excerpts in a toggle box
            with st.expander("📚 View source excerpts"):
                for i, excerpt in enumerate(context):
                    st.markdown(f"**Excerpt {i+1}:**")
                    st.markdown(excerpt)


# ==============================
# 🧠 Generate Pub Quiz
# ==============================
elif page == "Generate Pub Quiz":
    st.title("🧠 Generate a Pub Quiz")

    available_categories = controller.get_all_categories()
    if not available_categories:
        st.info("ℹ️ No categories ingested yet. Please ingest a some data first.")
    else:
        category = st.selectbox("Select a page:", options=available_categories)
        number_of_questions = st.number_input(
            "Number of questions", min_value=1, max_value=50, value=5, step=1
        )
        better_explanation = st.text_input(
            "What kind of questions would you like to focus on?",
            value="No specific kind.",
        )

        if st.button("🎲 Generate Pub Quiz"):
            with st.spinner("Selecting paragraphs and generating quiz..."):
                quiz = llm_controller.generate_quiz(
                    category, number_of_questions, lang_prefix, better_explanation
                )
                if quiz is None:
                    st.warning("Unable to generate quiz!")
                else:
                    for i, qa in enumerate(quiz, 1):
                        st.markdown(
                            f"**{difficulty_flag(qa['difficulty'])} Q{i}:** {qa['question']}"
                        )
                        with st.expander("Show Answer", expanded=True):
                            st.markdown(f"**A{i}:** {qa['answer']}")
                        with st.expander("Show Explanation", expanded=True):
                            st.markdown(f"**E{i}:** {qa['explanation']}")
                        st.markdown("---")
