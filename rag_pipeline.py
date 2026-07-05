from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import ChatOllama
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import EnsembleRetriever
import torch

def get_embeddings(model_name):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_llm():
    return ChatOllama(
        model="llama3.2",
        temperature=0
    )

def load_documents(pdf_paths):

    documents = []

    for pdf in pdf_paths:

        loader = PyPDFLoader(pdf)
        docs = loader.load()
        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

def build_vector_store(embedding, chunks, embedding_model):

    collection_name = embedding_model.replace("/", "_")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        collection_name=collection_name
    )
    return vector_store

def build_retriever(vector_store, chunks, strategy):

    if strategy == "Chroma MMR":

        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "fetch_k": 30
            }
        )

    bm25 = BM25Retriever.from_documents(chunks)

    bm25.k = 8

    chroma = vector_store.as_retriever(
        search_kwargs={"k": 8}
    )

    return EnsembleRetriever(
        retrievers=[bm25, chroma],
        weights=[0.4, 0.6]
    )

def build_chain(llm, retriever, prompt):

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt
        }
    )