# RAG LLM Demo

A Retrieval-Augmented Generation (RAG) system built with LangChain, ChromaDB, and Google Gemini for intelligent document querying.

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/chinsekyi-dev/rag-llm-demo
   cd rag-llm-demo
   ```

2. **Create a virtual environment**
   ```bash
   conda create -n dev python=3.13
   ```

3. **Install dependencies**
   ```bash
   make install
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GOOGLE_GENAI_API_KEY=your_gemini_api_key_here
   ```

5. **Add your PDF documents**
   Place PDF files in the `data/pdf/` directory

## Usage

Open the Jupyter notebook:
```bash
jupyter notebook notebooks/document.ipynb
```