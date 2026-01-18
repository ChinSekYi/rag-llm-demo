"""
Standalone RAG Runner - Test the RAG pipeline from command line
"""

import sys

import config
from rag_pipeline import create_rag_pipeline_from_scratch, load_rag_pipeline


def run_interactive():
    """Interactive mode - ask questions"""
    try:
        rag = load_rag_pipeline()
    except FileNotFoundError:
        rag = create_rag_pipeline_from_scratch()

    print("\n💬 Interactive Mode - Ask questions (type 'quit' to exit)")
    print("-" * 60)

    while True:
        question = input("\n❓ Question: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break

        if not question:
            continue

        result = rag.query_with_sources(question)


def run_test_queries():
    """Run predefined test queries"""
    try:
        rag = load_rag_pipeline()
    except FileNotFoundError:
        rag = create_rag_pipeline_from_scratch()

    print("\n" + "=" * 60)
    print("Running Test Queries")
    print("=" * 60)

    test_queries = [
        "What is deep learning?",
        "Explain attention mechanisms",
        "What are transformers?",
    ]

    for query in test_queries:
        rag.query_with_sources(query)
        print("\n" + "-" * 60)


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "create":
            print("Creating RAG pipeline from scratch...")
            create_rag_pipeline_from_scratch()
        elif sys.argv[1] == "test":
            print("Running test queries...")
            run_test_queries()
        elif sys.argv[1] == "interactive":
            run_interactive()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python run_rag.py [create|test|interactive]")
    else:
        # Default: interactive mode
        run_interactive()


if __name__ == "__main__":
    main()
