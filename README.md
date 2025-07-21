

Mini LLM-Powered Question-Answering System using RAG
====================================================

Objective
---------
This project implements a Retrieval-Augmented Generation (RAG) based question-answering system over a document. The system ingests a PDF, performs semantic chunking and embedding, stores it in a FAISS vector index, and answers user queries using an open-source LLM.

Components
----------

1. Document Ingestion & Chunking
- Loader Used: PyPDFLoader from langchain_community
- Splitter: RecursiveCharacterTextSplitter
  - chunk_size: 400 characters
  - chunk_overlap: 50 characters
  - Separation Strategy: ["\n\n", "\n", " ", ""] to preserve paragraph context

2. Embedding & Vector Store
- Model: sentence-transformers/all-MiniLM-L6-v2
- Framework: HuggingFaceEmbeddings
- Storage: FAISS

3. Query Interface
- Command-line interface using input()
- Accepts a user query and retrieves top-k relevant chunks

4. LLM Integration
- Model: vblagoje/bart_lfqa (HuggingFace)
- Pipeline: text2text-generation
- Prompt includes instruction to answer based on context or indicate no answer

5. Output
- Printed answer
- Source document and pages
- Time taken to generate answer

Tools Used
----------
- langchain: document loading and chunking
- transformers: LLM pipeline
- sentence-transformers: embedding generation
- FAISS: vector similarity search
- ChatGPT: used for architectural decisions and class structure

Sample Test Query & Output
--------------------------
Query:
  What are the diagnostic criteria for Obsessive-Compulsive Disorder (OCD)?

Output:
  Answer: [LLM-generated answer here]
  Source: book.pdf 
  <img width="1920" height="1080" alt="Screenshot (4)" src="https://github.com/user-attachments/assets/a5361e50-bdf2-4911-a430-443de186a9ed" />

  <img width="1920" height="1080" alt="Screenshot (4)" src="https://github.com/user-attachments/assets/87e43084-15db-45e9-8715-064239643da8" />

  <img width="1920" height="1080" alt="Screenshot (3)" src="https://github.com/user-attachments/assets/20e4ea1b-df25-4f22-ae3a-d3e8daec1e00" />



Completed Features
------------------
- PDF ingestion
- Chunking with overlap
- FAISS vector store
- HuggingFace LLM-based QA
- CLI interface
- Prompt engineering

Skipped / Incomplete
--------------------
- No GUI (Gradio/Streamlit)
- No caching for repeated queries
- No reranking logic
- CPU-only inference
- No dynamic document uploads

Design Assumptions
------------------
- Input is a clean, extractable-text PDF
- All models loadable on CPU
- Responses rely only on provided document context

How to Run
----------
```bash
pip install -r req.txt

python main.py
```
Use of ChatGPT
--------------
- Code structure and class design
- Prompt design
- Model selection
- README formatting

Conclusion
----------
This MVP provides an end-to-end RAG QA pipeline with clean modularity, extensibility, and full local execution support.
