import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import os
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, vector_store_path: str = None):
        self.vector_store_path = vector_store_path or "data/vector_store"
        self.documents_path = "data/documents"
        self.embeddings = None
        self.collection = None
        self.client = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv('CHUNK_SIZE', '500')),
            chunk_overlap=50
        )

    async def initialize(self):
        """Initialize the RAG service"""
        try:
            # Create directories if they don't exist
            os.makedirs(self.vector_store_path, exist_ok=True)
            os.makedirs(self.documents_path, exist_ok=True)
            os.makedirs("temp", exist_ok=True)

            # Initialize embeddings
            logger.info("Loading embeddings model...")
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            # Initialize ChromaDB with persistent storage
            self.client = chromadb.PersistentClient(
                path=self.vector_store_path,
                settings=Settings(allow_reset=True)
            )

            # Get or create collection
            try:
                self.collection = self.client.get_collection("documents")
                logger.info("Retrieved existing ChromaDB collection")
            except ValueError:
                self.collection = self.client.create_collection(
                    name="documents",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("Created new ChromaDB collection")

            logger.info("RAG service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {str(e)}")
            raise

    async def add_document(self, file_path: str) -> bool:
        """Add a document to the vector store."""
        try:
            # Load document based on file type
            loader = self._get_loader(file_path)
            if not loader:
                logger.error(f"Unsupported file type: {file_path}")
                return False
                
            documents = loader.load()
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            if not chunks:
                logger.warning(f"No content extracted from {file_path}")
                return False
            
            # Prepare data for ChromaDB
            texts = [chunk.page_content for chunk in chunks]
            metadatas = []
            ids = []
            
            filename = os.path.basename(file_path)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{filename}_{i}_{uuid.uuid4().hex[:8]}"
                ids.append(chunk_id)
                metadatas.append({
                    "filename": filename,
                    "chunk_index": i,
                    "source": file_path
                })
            
            # Generate embeddings and add to collection
            embeddings = [self.embeddings.embed_query(text) for text in texts]
            
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(chunks)} chunks from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {str(e)}")
            return False

    def _get_loader(self, file_path: str):
        """Get appropriate document loader based on file extension."""
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                return PyPDFLoader(file_path)
            elif file_extension in ['.txt', '.md']:
                return TextLoader(file_path)
            else:
                # Try TextLoader as fallback
                return TextLoader(file_path)
        except Exception as e:
            logger.error(f"Error creating loader for {file_path}: {e}")
            return None

    async def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        try:
            if not self.collection:
                logger.warning("RAG service not initialized")
                return []
                
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            # Format results
            documents = []
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    score = results['distances'][0][i] if results['distances'] and results['distances'][0] else 0.0
                    documents.append({
                        'content': doc,
                        'metadata': metadata,
                        'score': score
                    })
            
            logger.info(f"Found {len(documents)} relevant documents for query: {query[:50]}...")
            return documents
            
        except Exception as e:
            logger.error(f"Document search error: {str(e)}")
            return []

    def list_documents(self) -> List[str]:
        """List all documents in the collection."""
        try:
            if not os.path.exists(self.documents_path):
                return []
            
            documents = []
            for file in os.listdir(self.documents_path):
                if os.path.isfile(os.path.join(self.documents_path, file)):
                    documents.append(file)
            
            return documents
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []

    async def delete_document(self, filename: str) -> bool:
        """Delete a document from the vector store."""
        try:
            if not self.collection:
                return False
                
            # Get all document IDs that match the filename
            results = self.collection.get(
                where={"filename": filename}
            )
            
            if results and results['ids']:
                # Delete all chunks for this document
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {filename}")
                return True
            else:
                logger.warning(f"No chunks found for document {filename}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document {filename}: {str(e)}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            if not self.collection:
                return {"total_chunks": 0, "status": "not_initialized"}
                
            count = self.collection.count()
            return {
                "total_chunks": count,
                "status": "active"
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                "total_chunks": 0,
                "status": "error"
            }

    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            if not self.collection:
                return
                
            # Get all IDs
            results = self.collection.get()
            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info("Cleared all documents from collection")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
