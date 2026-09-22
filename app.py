import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="Ask PDF", page_icon="📄", layout="wide")
st.title("📄 Ask PDF")
st.caption("Basic RAG: PDF → Chunk → Embed → Retrieve → Answer")


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


def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        pdf_path = temp_file.name

    try:
        documents = PyMuPDFLoader(pdf_path).load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        vector_db = FAISS.from_documents(chunks, get_embeddings())
        return vector_db, len(documents), len(chunks)
    finally:
        os.remove(pdf_path)


def retrieve_and_answer(vector_db, question):
    retrieved_docs = vector_db.similarity_search(question, k=4)

    context_parts = []
    for doc in retrieved_docs:
        page_number = doc.metadata.get("page", 0) + 1
        context_parts.append(f"[Page {page_number}]\n{doc.page_content}")

    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages([
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
        ("human", "{question}"),
    ])

    response = (prompt | get_llm()).invoke({
        "context": context,
        "question": question,
    })

    return response.content, retrieved_docs


if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "pdf_id" not in st.session_state:
    st.session_state.pdf_id = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "page_count" not in st.session_state:
    st.session_state.page_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

with st.sidebar:
    st.header("PDF")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        pdf_id = f"{uploaded_file.name}:{uploaded_file.size}"

        if st.session_state.pdf_id != pdf_id:
            with st.spinner("Processing PDF..."):
                vector_db, page_count, chunk_count = process_pdf(uploaded_file)

            st.session_state.vector_db = vector_db
            st.session_state.pdf_id = pdf_id
            st.session_state.pdf_name = uploaded_file.name
            st.session_state.page_count = page_count
            st.session_state.chunk_count = chunk_count
            st.session_state.messages = []
            st.success("PDF is ready!")

    st.divider()

    if st.session_state.vector_db is not None:
        st.subheader("Document Info")
        st.write(f"**File:** {st.session_state.pdf_name}")
        st.write(f"**Pages:** {st.session_state.page_count}")
        st.write(f"**Chunks:** {st.session_state.chunk_count}")
        st.write("**Retrieval:** Top 4 chunks")
        st.write("**LLM:** Gemini 2.5 Flash")
        st.write("**Embeddings:** Gemini Embedding 2 Preview")

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about your PDF...")

if question:
    if st.session_state.vector_db is None:
        st.warning("Please upload a PDF first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PDF..."):
            answer, retrieved_docs = retrieve_and_answer(
                st.session_state.vector_db, question
            )
        st.markdown(answer)

        pages = sorted({doc.metadata.get("page", 0) + 1 for doc in retrieved_docs})
        if pages:
            st.caption("Retrieved pages: " + ", ".join(map(str, pages)))

    st.session_state.messages.append({"role": "assistant", "content": answer})
