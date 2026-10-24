"""Chunk -> embed -> FAISS vector database (same pipeline as the Research Paper project)."""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import EMBEDDING_MODEL


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vector_store(chunks: list[Document], embeddings) -> FAISS:
    if not chunks:
        raise ValueError("No text to index. Add brand documents to the Knowledge folder.")
    return FAISS.from_documents(chunks, embeddings)


def save_store(store: FAISS, folder: str = "vector_db") -> None:
    store.save_local(folder)


def load_store(embeddings, folder: str = "vector_db") -> FAISS:
    # Only load an index that YOU created: pickle files can run code.
    return FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
