import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import warnings
warnings.filterwarnings("ignore")
class DocumentProcessor:
    @staticmethod
    def load_and_chunk(file_path="book.pdf", chunk_size=400, chunk_overlap=50):
        """Load and chunk PDF document with better text preservation"""
        print(f"Processing document: {os.path.basename(file_path)}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # Improved splitting that better preserves context
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""]  # Better paragraph handling
        )
        chunks = text_splitter.split_documents(docs)
        print(f"Created {len(chunks)} chunks")
        return chunks

class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        self.vector_db = None
        
    def create_store(self, chunks, index_name="faiss_index"):
        print("Creating vector store...")
        self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        self.vector_db.save_local(index_name)
        return self.vector_db
    
    def load_store(self, index_name="faiss_index"):
        print("Loading vector store...")
        self.vector_db = FAISS.load_local(index_name, self.embeddings, allow_dangerous_deserialization=True)
        return self.vector_db

class QAEngine:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        print("Loading improved QA model...")
        
        # Using a model better suited for document QA
        model_name = "vblagoje/bart_lfqa"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        self.qa_pipeline = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1  # Force CPU
        )
    
    def generate_answer(self, query, k=3):
        """Generate answer with better context handling"""
        start_time = time.time()
        
        # Retrieve more context chunks
        context_chunks = self.vector_db.similarity_search(query, k=k)
        context_text = "\n".join([
            f"Document excerpt {i+1} (from {chunk.metadata.get('source', 'Unknown')} page {chunk.metadata.get('page', 'N/A')}):\n"
            f"{chunk.page_content}\n" 
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Improved prompt engineering
        prompt = (
            "Read the following document excerpts and answer the question. "
            "If the answer cannot be found in the documents, say 'I cannot find the answer in the documents'.\n\n"
            f"Documents:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        
        # Generate with better parameters
        result = self.qa_pipeline(
            prompt,
            max_length=200,
            num_beams=5,  # Better quality than greedy search
            temperature=0.7,
            early_stopping=True
        )
        
        return {
            "answer": result[0]['generated_text'].strip(),
            "sources": list(set(chunk.metadata.get('source') for chunk in context_chunks)),
            "time": time.time() - start_time
        }

def main():
    # Configuration
    DOCUMENT_PATH = "book.pdf"
    VECTOR_INDEX = "faiss_index"
    
    # Document processing
    if not os.path.exists(VECTOR_INDEX):
        chunks = DocumentProcessor.load_and_chunk(DOCUMENT_PATH)
        vector_db = VectorStoreManager().create_store(chunks, VECTOR_INDEX)
    else:
        vector_db = VectorStoreManager().load_store(VECTOR_INDEX)
    
    # Initialize QA system
    qa_engine = QAEngine(vector_db)
    
    # Interactive mode
    print("\nEnter questions about the document (Ctrl+C to exit):")
    while True:
        try:
            query = input("\nQuestion: ").strip()
            if not query:
                continue
                
            result = qa_engine.generate_answer(query)
            print(f"\nAnswer: {result['answer']}")
            print(f"Source: {', '.join(result['sources'])}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()