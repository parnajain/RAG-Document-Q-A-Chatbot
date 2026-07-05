from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
import streamlit as st

# Load API Keys and Environment Variables
load_dotenv()

from rag_pipeline import (
    get_embeddings,
    get_llm,
    load_documents,
    build_vector_store,
    build_retriever,
    build_chain
)
# Streamlit UI
st.title("📄 Your Personal Chatbot")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# Strategy Selector
strategy = st.radio(
    "Select RAG Strategy",
    ["Chroma MMR", "Ensemble BM25 + Chroma"],
    horizontal=True
)

embedding_model = st.selectbox(
    "Embedding Model",
    ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5"]
)
# Prompt Template
prompt_template = """
You are a document assistant.

RULES:
1. Answer only using the provided context.
2. If the user asks for a summary, create a summary from the context.
3. If the answer cannot be found in the context, respond:

Out of scope: The answer is not present in the uploaded document.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

@st.cache_resource
def cached_embeddings(model_name):
    return get_embeddings(model_name)

@st.cache_resource
def cached_llm():
    return get_llm()

llm = cached_llm()
embedding = cached_embeddings(embedding_model)
# -------------------------
# Process PDF
# -------------------------
if uploaded_files:

    current_state = (
        tuple(sorted((f.name, f.size) for f in uploaded_files)),
        embedding_model
    )
    
    # Rebuild only when files change
    if (
        "loaded_files" not in st.session_state
        or st.session_state.loaded_files != current_state
    ):
        print("Processing PDFs...")

        pdf_paths = []

        for uploaded_file in uploaded_files:

            temp_path = f"temp_{uploaded_file.name}"

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            pdf_paths.append(temp_path)

        chunks = load_documents(pdf_paths)
        
        for path in pdf_paths:
            if os.path.exists(path):
                os.remove(path)

    # creating a vector store using Chroma for efficient similarity search and retrieval. The embeddings are generated using a 
    # HuggingFace model, which converts the text chunks into vector representations that can be easily compared with user queries.
        vector_store = build_vector_store(
            embedding,
            chunks,
            embedding_model
        )

        st.session_state.vector_store = vector_store
        st.session_state.chunks = chunks
        st.session_state.loaded_files = current_state

    vector_store = st.session_state.vector_store
    chunks = st.session_state.chunks

    retriever = build_retriever(
        vector_store,
        chunks,
        strategy
    )

    qa_chain = build_chain(
        llm,
        retriever,
        PROMPT
    )
    st.success(f"Document Processed! Using: **{strategy}**")

    query = st.text_input(
        "Ask a question about the document"
    )

    if query:

        summary_keywords = ["summary", "summarize", "summarise", "overview", "brief", "what is this document about"]
        is_summary_request = any(kw in query.lower() for kw in summary_keywords)

        docs_with_scores = vector_store.similarity_search_with_score(query, k=8)
        best_score = docs_with_scores[0][1]
        print("Best Score:", best_score)

        if best_score > 0.9 and not is_summary_request:
            st.subheader("Answer")
            st.write("Out of scope: The answer is not present in the uploaded document.")

        else:
            result = qa_chain.invoke({"query": query})
            answer = result["result"]
            source_docs = result["source_documents"]

            st.subheader("Answer")
            st.write(answer)

            with st.expander("Sources"):
                for doc in source_docs:
                    st.write(doc.page_content[:500])
