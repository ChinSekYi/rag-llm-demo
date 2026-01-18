"""
Debug script to inspect the vector database
"""

import os

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import config


def inspect_vector_db(chroma_path=config.CHROMA_DB_DIR):
    """Inspect the contents of the vector database"""
    print("=" * 60)
    print("🔍 Vector Database Inspector")
    print("=" * 60)

    # Check if path exists
    if not os.path.exists(chroma_path):
        print(f"❌ Chroma DB not found at: {chroma_path}")
        print(f"   Absolute path: {os.path.abspath(chroma_path)}")
        print("\nTo create it, run: python run_rag.py create")
        return

    print(f"✅ Chroma DB found at: {os.path.abspath(chroma_path)}")

    # Check directory contents
    print(f"\n📁 Directory contents:")
    for item in os.listdir(chroma_path):
        item_path = os.path.join(chroma_path, item)
        size = os.path.getsize(item_path)
        print(f"   - {item} ({size} bytes)")

    # Try to load the vector store
    try:
        print(f"\n🔗 Loading embeddings model: {config.EMBEDDINGS_MODEL}")
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDINGS_MODEL)

        print(f"📂 Loading vector store...")
        vector_store = Chroma(
            persist_directory=chroma_path, embedding_function=embeddings
        )

        # Check collection stats
        all_docs = vector_store.get()
        num_docs = len(all_docs["ids"])
        print(f"\n📊 Vector Store Stats:")
        print(f"   Total documents: {num_docs}")

        if num_docs == 0:
            print("\n⚠️  Vector store is EMPTY!")
            print("   Run: python run_rag.py create")
            return

        # Show sample documents
        print(f"\n📄 Sample Documents:")
        for i in range(min(5, num_docs)):
            doc_id = all_docs["ids"][i]
            metadata = all_docs["metadatas"][i]
            content = all_docs["documents"][i][:100] if all_docs["documents"] else "N/A"

            print(f"\n   Doc {i+1}:")
            print(f"      ID: {doc_id}")
            print(f"      Source: {metadata.get('source', 'Unknown')}")
            print(f"      Page: {metadata.get('page', 'Unknown')}")
            print(f"      Content preview: {content}...")

        # Test retrieval
        print(f"\n🧪 Testing Retrieval:")
        test_queries = [
            "deep learning",
            "machine learning",
            "neural networks",
            "What is AI?",
        ]

        for query in test_queries:
            retriever = vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}
            )
            results = retriever.invoke(query)
            print(f"\n   Query: '{query}'")
            print(f"   Results: {len(results)} documents")
            if results:
                for j, doc in enumerate(results, 1):
                    source = doc.metadata.get("source", "Unknown")
                    print(f"      {j}. {source}")
            else:
                print(f"      ⚠️  No results found!")

    except Exception as e:
        print(f"\n❌ Error loading vector store: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    inspect_vector_db()
