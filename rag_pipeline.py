"""
RAG Pipeline Module - Handles document loading, embedding, and retrieval
"""

import os

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


class DocumentLoader:
    """Load documents from PDF directory"""

    @staticmethod
    def load_pdfs(pdf_dir: str):
        """Load all PDFs from directory"""
        print(f"Loading PDFs from: {pdf_dir}")
        dir_loader = DirectoryLoader(
            pdf_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader, show_progress=True
        )
        documents = dir_loader.load()
        print(f"✅ Loaded {len(documents)} PDF pages")
        return documents


class TextSplitter:
    """Split documents into chunks"""

    @staticmethod
    def split(
        documents, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    ):
        """Split documents into chunks"""
        print(f"Splitting documents (chunk_size={chunk_size}, overlap={chunk_overlap})")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=config.SEPARATORS,
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"✅ Split into {len(split_docs)} chunks")
        return split_docs


class VectorStoreManager:
    """Manage vector store creation and persistence"""

    @staticmethod
    def create_vector_store(
        documents,
        embeddings_model=config.EMBEDDINGS_MODEL,
        persist_dir=config.CHROMA_DB_DIR,
    ):
        """Create and persist vector store"""
        print(f"Creating embeddings with model: {embeddings_model}")
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)

        print(f"Creating vector store at: {persist_dir}")
        vector_store = Chroma.from_documents(
            documents=documents, embedding=embeddings, persist_directory=persist_dir
        )
        print(f"✅ Vector store created with {len(documents)} documents")
        return vector_store, embeddings

    @staticmethod
    def load_vector_store(
        embeddings_model=config.EMBEDDINGS_MODEL, persist_dir=config.CHROMA_DB_DIR
    ):
        """Load existing vector store"""
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"Vector store not found at: {persist_dir}")

        print(f"Loading vector store from: {persist_dir}")
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        vector_store = Chroma(
            persist_directory=persist_dir, embedding_function=embeddings
        )
        print(f"✅ Vector store loaded")
        return vector_store, embeddings


class LLMManager:
    """Manage LLM initialization"""

    @staticmethod
    def create_ollama_llm(model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE):
        print(f"Initializing Ollama LLM: {model}")
        llm = ChatOllama(
            model=model, temperature=temperature, base_url=config.LLM_BASE_URL
        )
        print(f"✅ LLM initialized")
        return llm


class RAGPipeline:
    """Main RAG Pipeline - combines all components"""

    def __init__(self, vector_store, retriever, llm):
        self.vector_store = vector_store
        self.retriever = retriever
        self.llm = llm
        self.rag_chain = self._build_chain()

    def _build_chain(self):
        """Build the RAG chain"""
        print("Building RAG chain...")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", config.SYSTEM_PROMPT),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print("✅ RAG chain built")
        return rag_chain

    def query(self, question: str):
        """Query the RAG system"""
        print(f"\n🔍 Query: {question}")
        answer = self.rag_chain.invoke(question)
        retrieved_docs = self.retriever.invoke(question)

        return {"question": question, "answer": answer, "sources": retrieved_docs}

    def query_with_sources(self, question: str):
        """Query and display with sources"""
        result = self.query(question)

        print(f"\n📝 Answer:")
        print(result["answer"])

        print(f"\n📖 Retrieved Sources ({len(result['sources'])} docs):")
        for i, doc in enumerate(result["sources"], 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            print(f"  {i}. {source} (page {page})")

        return result


def create_rag_pipeline_from_scratch(pdf_dir=config.DATA_PDF_DIR):
    """Create RAG pipeline from scratch (load PDFs, create embeddings)"""
    print("=" * 60)
    print("Creating RAG Pipeline from Scratch")
    print("=" * 60)

    # Load documents
    documents = DocumentLoader.load_pdfs(pdf_dir)

    # Split documents
    split_docs = TextSplitter.split(documents)

    # Create vector store
    vector_store, embeddings = VectorStoreManager.create_vector_store(split_docs)

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type=config.RETRIEVER_SEARCH_TYPE,
        search_kwargs={"k": config.RETRIEVER_K},
    )

    # Create LLM
    llm = LLMManager.create_ollama_llm()

    # Build RAG pipeline
    rag_pipeline = RAGPipeline(vector_store, retriever, llm)

    print("=" * 60)
    print("✅ RAG Pipeline Ready!")
    print("=" * 60)

    return rag_pipeline


def load_rag_pipeline():
    """Load existing RAG pipeline (use saved vector store)"""
    print("=" * 60)
    print("Loading RAG Pipeline")
    print("=" * 60)

    # Load vector store
    vector_store, embeddings = VectorStoreManager.load_vector_store()

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type=config.RETRIEVER_SEARCH_TYPE,
        search_kwargs={"k": config.RETRIEVER_K},
    )

    # Create LLM
    llm = LLMManager.create_ollama_llm()

    # Build RAG pipeline
    rag_pipeline = RAGPipeline(vector_store, retriever, llm)

    print("=" * 60)
    print("✅ RAG Pipeline Loaded!")
    print("=" * 60)

    return rag_pipeline
