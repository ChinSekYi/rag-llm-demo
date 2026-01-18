import os

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Page config
st.set_page_config(page_title="RAG Chat with Ollama", layout="wide")
st.title("📚 RAG Chat with Ollama")


@st.cache_resource
def initialize_rag():
    """Initialize RAG system once and cache it"""
    try:
        # Get the correct path to chroma_db
        # If running from project root with: streamlit run app_interface/app.py
        chroma_path = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        chroma_path = os.path.abspath(chroma_path)

        st.write(f"🔍 Using chroma_db at: {chroma_path}")

        if not os.path.exists(chroma_path):
            st.error(f"❌ chroma_db not found at: {chroma_path}")
            st.info("Please run the ingestion notebook first to create embeddings.")
            return None, None

        # Load embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma(
            persist_directory=chroma_path, embedding_function=embeddings
        )
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

        # Initialize Ollama LLM
        llm = ChatOllama(model="gpt-oss:20b", temperature=0.7)

        # Create RAG chain
        system_prompt = """You are an AI assistant that answers questions based on the provided context. 
If the context doesn't contain the answer, say "I don't have enough information to answer this question."
Keep your answers concise and relevant."""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Context:\n{context}\n\nQuestion: {question}"),
            ]
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain, retriever

    except Exception as e:
        st.error(f"❌ Error initializing RAG: {str(e)}")
        st.info(
            "Make sure: 1) Ollama is running (`ollama serve`) 2) Ingestion notebook was executed"
        )
        return None, None


# Initialize RAG system
with st.spinner("Loading RAG system..."):
    rag_chain, retriever = initialize_rag()

if rag_chain and retriever:
    st.success("✅ System Ready")

    # Query input
    user_query = st.text_input(
        "Ask a question about your documents:",
        placeholder="e.g., What is deep learning?",
    )

    if user_query:
        with st.spinner("Searching documents and generating answer..."):
            try:
                # Get answer
                answer = rag_chain.invoke(user_query)

                # Get retrieved documents
                retrieved_docs = retriever.invoke(user_query)

                # Display answer
                st.subheader("📝 Answer")
                st.write(answer)

                # Display sources
                st.subheader("📖 Retrieved Sources")
                if retrieved_docs:
                    for i, doc in enumerate(retrieved_docs, 1):
                        with st.expander(
                            f"Source {i}: {doc.metadata.get('source', 'Unknown')}"
                        ):
                            st.write(doc.page_content[:500] + "...")
                            st.caption(f"Page: {doc.metadata.get('page', 'N/A')}")
                else:
                    st.warning("No documents retrieved for this query")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

else:
    st.warning("⚠️ Failed to initialize RAG system. Check the error messages above.")
