"""
RAG Vector Store for MIE Knowledge Base
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import pickle
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
except ImportError:
    print("Installing required packages...")
    os.system("pip install sentence-transformers scikit-learn numpy")

class MIEVectorStore:
    """Vector store for MIE knowledge base using sentence transformers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = SentenceTransformer(config["rag"]["embedding_model"])
        self.vector_store_path = Path(config["rag"]["vector_store_path"])
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
    def train_with_mie_data(self, data_path: str = None):
        """Train the RAG system with MIE dataset"""
        if data_path is None:
            data_path = "final_data_true.csv"
            
        print(f"Training RAG with MIE data from {data_path}...")
        
        # Load dataset
        df = pd.read_csv(data_path)

        # Flexible column detection to support both master and mie_only CSVs
        def _pick(colnames):
            for name in colnames:
                if name in df.columns:
                    return name
            return None

        title_col = _pick(["Title", "title"])
        subject_col = _pick(["Subject ", "Subject", "subject"])
        text_col = _pick(["Text", "text", "content", "body"])  # fallbacks
        label_col = _pick(["Probable MIE", "probable", "label", "is_mie"])  # fallbacks

        # Build label column robustly
        if label_col is not None:
            vals = df[label_col]
            if vals.dtype.kind in {"i", "u", "f"}:
                df['label'] = (vals.astype(float) >= 1.0).astype(int)
            else:
                df['label'] = vals.astype(str).str.lower().isin(["1", "true", "yes", "mie"]).astype(int)
        else:
            df['label'] = 0
        
        # Create combined text
        parts = []
        if title_col is not None:
            parts.append(df[title_col].fillna(''))
        if subject_col is not None:
            parts.append(df[subject_col].fillna(''))
        if text_col is not None:
            parts.append(df[text_col].fillna(''))
        if not parts:
            raise ValueError("No suitable text columns found. Expected one of Title/title and Text/text.")
        df['combined_text'] = (parts[0] if len(parts) == 1 else parts[0].astype(str))
        for p in parts[1:]:
            df['combined_text'] = df['combined_text'].astype(str) + ' ' + p.astype(str)
        
        # Add articles to vector store
        documents = df['combined_text'].tolist()
        metadata = []
        
        for idx, row in df.iterrows():
            metadata.append({
                "source": "mie_dataset",
                "type": "article",
                "article_id": idx,
                "title": row[title_col] if title_col is not None else "",
                "label": int(row['label']),
                "mie_probable": (row[label_col] if label_col is not None else 0)
            })
        
        self.add_documents(documents, metadata)
        self.save()
        
        print(f"✅ RAG trained with {len(documents)} MIE articles")
        
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the vector store"""
        if metadata is None:
            metadata = [{"source": f"doc_{i}"} for i in range(len(documents))]
            
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents, show_progress_bar=True)
        
        # Store
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.metadata.extend(metadata)
        
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if not self.documents:
            return []
            
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "metadata": self.metadata[idx],
                "similarity": float(similarities[idx])
            })
            
        return results
    
    def save(self):
        """Save vector store to disk"""
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings,
            "metadata": self.metadata
        }
        
        with open(self.vector_store_path / "vectorstore.pkl", "wb") as f:
            pickle.dump(data, f)
    
    def load(self):
        """Load vector store from disk"""
        vectorstore_file = self.vector_store_path / "vectorstore.pkl"
        if vectorstore_file.exists():
            with open(vectorstore_file, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.embeddings = data["embeddings"]
                self.metadata = data["metadata"] 