import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Ask PDF",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Ask PDF")
st.caption("Basic RAG: PDF → Chunk → Embed → Retrieve → Answer")

# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL")
    )

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("LLM_MODEL")
    )

# ---------------------------------------------------------
# PDF PROCESSING
# ---------------------------------------------------------

def process_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(
        delete=False, 
        suffix=".pdf"
        ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())
        pdf_path = temp_file.name

    try:

        # Load PDF
        documents = PyMuPDFLoader(pdf_path).load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        # Create vector database
        vector_db = FAISS.from_documents(chunks, get_embeddings())

        return vector_db, len(documents), len(chunks)

    finally:
        os.remove(pdf_path)


# ---------------------------------------------------------
# RESPONSE TEXT EXTRACTION
# ---------------------------------------------------------

def extract_response_text(response):
    content = response.content

    # Normal string response
    if isinstance(content, str):
        return content

    # Gemini/LangChain structured response
    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                if "text" in item:
                    text_parts.append(item["text"])

        return "\n".join(text_parts)

    return str(content)


# ---------------------------------------------------------
# RETRIEVAL + ANSWERING
# ---------------------------------------------------------

def retrieve_and_answer(vector_db, question):

    # Retrieve top 4 chunks
    retrieved_docs = vector_db.similarity_search(question, k=4)

    context_parts = []

    for doc in retrieved_docs:

        page_number = doc.metadata.get("page", 0) + 1

        context_parts.append(
            f"[Page {page_number}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the information contained in the retrieved PDF context below.

If the answer is not present in the context, say:
"I couldn't find that information in the PDF."

Do not invent or assume information.

Retrieved PDF context:
{context}""",
            ),
            (
                "human",
                "{question}"
            ),
        ]
    )

    # Call LLM
    response = (prompt | get_llm()).invoke(
        {
            "context": context,
            "question": question,
        }
    )

    # Extract actual text from Gemini response
    answer = extract_response_text(response)

    return answer, retrieved_docs


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "pdf_id" not in st.session_state:
    st.session_state.pdf_id = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

if "page_count" not in st.session_state:
    st.session_state.page_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "show_pdf" not in st.session_state:
    st.session_state.show_pdf = True

if "show_ask" not in st.session_state:
    st.session_state.show_ask = True

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("PDF")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        pdf_id = f"{uploaded_file.name}:{uploaded_file.size}"

        if st.session_state.pdf_id != pdf_id:

            with st.spinner("Processing PDF..."):

                vector_db, page_count, chunk_count = process_pdf(uploaded_file)

            # Store PDF data
            st.session_state.vector_db = vector_db
            st.session_state.pdf_id = pdf_id
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.pdf_bytes = uploaded_file.getvalue()
            st.session_state.page_count = page_count
            st.session_state.chunk_count = chunk_count

            # Reset chat
            st.session_state.messages = []

            # Open both sections
            st.session_state.show_pdf = True
            st.session_state.show_ask = True

            st.success("PDF is ready!")

    st.divider()

    # Document information
    if st.session_state.vector_db is not None:

        st.subheader("Document Info")

        st.write(f"**File:** {st.session_state.pdf_name}")

        st.write(f"**Pages:** {st.session_state.page_count}")

        st.write(f"**Chunks:** {st.session_state.chunk_count}")

        st.write("**Retrieval:** Top 4 chunks")

        st.write(f"**LLM:** {os.getenv('LLM_MODEL')}")

        st.write(f"**Embeddings:** {os.getenv('EMBEDDING_MODEL')}")

    st.divider()

    # Clear chat
    if st.button(
        "Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ---------------------------------------------------------
# REOPEN BUTTONS
# ---------------------------------------------------------

if (
    st.session_state.vector_db is not None
    and (
        not st.session_state.show_pdf
        or not st.session_state.show_ask
    )
):

    reopen_columns = st.columns(2)

    with reopen_columns[0]:

        if not st.session_state.show_pdf:

            if st.button(
                "📄 Open PDF Viewer",
                use_container_width=True
            ):

                st.session_state.show_pdf = True

                st.rerun()

    with reopen_columns[1]:

        if not st.session_state.show_ask:

            if st.button(
                "💬 Open Ask Section",
                use_container_width=True
            ):

                st.session_state.show_ask = True

                st.rerun()

    st.divider()


# ---------------------------------------------------------
# PDF VIEWER
# ---------------------------------------------------------

def display_pdf(pdf_bytes):

    if not pdf_bytes:
        st.info("Upload a PDF to view it here.")
        return

    # Native Streamlit PDF viewer.
    # This avoids the Base64 data: iframe that Brave may block.
    st.pdf(pdf_bytes, height=750)


# ---------------------------------------------------------
# MAIN CONTENT
# ---------------------------------------------------------

if (
    st.session_state.show_pdf
    and st.session_state.show_ask
):

    # Both sections open
    pdf_column, ask_column = st.columns(2, gap="medium")

elif st.session_state.show_pdf:

    # Only PDF section open
    pdf_column = st.container()

elif st.session_state.show_ask:

    # Only Ask section open
    ask_column = st.container()


# ---------------------------------------------------------
# PDF SECTION
# ---------------------------------------------------------

if st.session_state.show_pdf:

    with pdf_column:
        header_column, close_column = st.columns([5, 1])

        with header_column:
            st.subheader("📄 PDF Viewer")

        with close_column:
            if st.button(
                "✕ Close",
                key="close_pdf",
                use_container_width=True
            ):

                st.session_state.show_pdf = False

                st.rerun()

        if st.session_state.pdf_bytes:

            display_pdf(st.session_state.pdf_bytes)

        else:
            st.info("Upload a PDF from the sidebar to view it here.")

# ---------------------------------------------------------
# ASK SECTION
# ---------------------------------------------------------

if st.session_state.show_ask:

    with ask_column:

        header_column, close_column = st.columns([5, 1])

        with header_column:
            st.subheader("💬 Ask Questions")

        with close_column:
            if st.button(
                "✕ Close",
                key="close_ask",
                use_container_width=True
            ):

                st.session_state.show_ask = False

                st.rerun()

        # Display previous messages
        for message in st.session_state.messages:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Question input
        question = st.chat_input("Ask a question about your PDF...")

        if question:
            if st.session_state.vector_db is None:
                st.warning("Please upload a PDF first.")

                st.stop()

            # Add user question to history
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user"):
                st.markdown(question)

            # Generate answer
            with st.chat_message("assistant"):

                with st.spinner("Searching the PDF..."):

                    answer, retrieved_docs = retrieve_and_answer(
                        st.session_state.vector_db,
                        question
                    )

                st.markdown(answer)

                # Show retrieved pages
                pages = sorted(
                    {
                        doc.metadata.get("page", 0) + 1
                        for doc in retrieved_docs
                    }
                )

                if pages:

                    st.caption("Retrieved pages: "+ ", ".join(map(str, pages)))

            # Save assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )
