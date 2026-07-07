
 Medical QA System with RAG

A production-style Medical Question Answering system utilizing Retrieval-Augmented Generation (RAG). The application allows users to upload medical PDFs, automatically extracts and embeds the text, and answers medical queries strictly using the retrieved context to eliminate hallucinations.

Features :
PDF Upload & Processing: High-quality text extraction with OCR fallback for scanned medical images.
  Smart Chunking : Splits large documents into manageable sizes while retaining contextual overlap.
  Two-Stage Retrieval : Uses local HuggingFace embeddings for fast broad-search, followed by a semantic Cross-Encoder to re-rank the absolute best results.
  Strict Prompting : Enforces the LLM to only answer from the provided context to prevent dangerous medical hallucinations.
  Citations & Confidence : The AI outputs exact text citations to back up claims and provides a confidence score.
  Evaluation Dashboard : Built-in Admin panel using the `ragas` framework to test Faithfulness, Context Precision, and Answer Relevance.

 Technical Architecture

1. Frontend: Streamlit
The entire user interface is built using Streamlit (`app.py`). It features a custom glassmorphism dark-mode UI, a conversational chat interface, and a dual-tab layout separating the Chat/QA functionality from the Admin Evaluation panel.

 2. Document Processing & Ingestion (`src/document_processor.py`)
PDF Parsing : Uses `pdfplumber`.
OCR Fallback : Uses `pdf2image` and `pytesseract` (Tesseract OCR).
Chunking : Uses LangChain's `RecursiveCharacterTextSplitter`.

3. Vector Database & Embeddings (`src/vector_store.py`)
Embeddings : Uses `HuggingFaceEmbeddings` with the `all-MiniLM-L6-v2` model.
Vector Store : Uses `ChromaDB` running locally and persistently to store and query the document vectors.

 4. Advanced Retrieval Pipeline (`src/retriever.py`)
First Stage : Retrieves top chunks from ChromaDB.
Second Stage (Reranking) : Uses a HuggingFace Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-rank the retrieved chunks based on deep semantic relevance.

6. Generative QA Chain (`src/qa_chain.py`)
LLM Integration : Supports both Google Gemini (`gemini-1.5-pro`) and OpenAI (`gpt-4o`) via LangChain.
Structured Output : Uses `JsonOutputParser` to force the LLM to return a strict JSON format (Answer, Citations, Confidence, Safety).
