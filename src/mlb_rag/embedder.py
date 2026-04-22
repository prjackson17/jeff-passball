"""
embedder.py

Embeds MLBChunk objects using a HuggingFace sentence transformer
and indexes them in a FAISS vector store for semantic retrieval.

This is the core neural network component of the RAG pipeline.
The sentence transformer (all-MiniLM-L6-v2) is a 6-layer transformer
that produces 384-dim dense embeddings. Retrieval uses cosine similarity
over the embedding space.

Author: Parker Jackson
Course: CSCI 357 - AI and Neural Networks
"""

import numpy as np
import faiss
import torch
import pickle
import os
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer

from src.mlb_rag.data_ingestion import MLBChunk


# ── Config ─────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384          # output dim of all-MiniLM-L6-v2
INDEX_PATH = "./mlb_index.faiss"
CHUNKS_PATH = "./mlb_chunks.pkl"


# ── Embedder ───────────────────────────────────────────────────────────────────

class MLBEmbedder:
    """
    Wraps a HuggingFace sentence transformer for encoding MLB text chunks.

    The model maps variable-length text → fixed 384-dim vector via:
        1. WordPiece tokenization
        2. 6-layer transformer encoder
        3. Mean pooling over token embeddings
        4. L2 normalization (for cosine similarity via dot product)

    We use a pretrained model here. The neural network contribution is:
    - Understanding WHY this architecture works for semantic retrieval
    - Chunking strategy decisions (what text goes in, how it's structured)
    - Retrieval design (top-k, similarity threshold, reranking)
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"[Embedder] Loading {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"[Embedder] Model loaded. Embedding dim: {self.model.get_sentence_embedding_dimension()}")

    def embed(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """
        Embed a list of strings → (N, embedding_dim) float32 numpy array.

        Args:
            texts: List of text strings to embed.
            batch_size: How many texts to encode per forward pass.
            show_progress: Show tqdm progress bar.

        Returns:
            np.ndarray of shape (N, 384), normalized to unit length.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # L2 normalize → cosine sim = dot product
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_chunks(self, chunks: List[MLBChunk], **kwargs) -> np.ndarray:
        """Convenience method: embed a list of MLBChunks by their .text field."""
        texts = [chunk.text for chunk in chunks]
        return self.embed(texts, **kwargs)


# ── FAISS Vector Store ─────────────────────────────────────────────────────────

class MLBVectorStore:
    """
    FAISS-backed vector store for MLB chunk embeddings.

    Uses IndexFlatIP (inner product) which, combined with L2-normalized
    embeddings from the embedder, gives exact cosine similarity search.

    For larger corpora, swap to IndexIVFFlat for approximate nearest
    neighbor search (faster but slightly less accurate).
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        # IndexFlatIP: brute-force inner product (= cosine sim for normed vecs)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[MLBChunk] = []    # parallel list to FAISS index rows

    def add(self, chunks: List[MLBChunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the store.

        Args:
            chunks: List of MLBChunk objects (metadata + text).
            embeddings: np.ndarray of shape (N, embedding_dim).
        """
        assert len(chunks) == embeddings.shape[0], "chunks and embeddings must have same length"
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        print(f"[VectorStore] Added {len(chunks)} chunks. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[MLBChunk, float]]:
        """
        Retrieve the top-k most similar chunks for a query embedding.

        Args:
            query_embedding: Shape (1, embedding_dim) or (embedding_dim,).
            top_k: Number of results to return.

        Returns:
            List of (MLBChunk, similarity_score) tuples, sorted by score descending.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Clamp top_k to available items
        top_k = min(top_k, self.index.ntotal)
        if top_k == 0:
            return []

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:    # FAISS returns -1 for empty slots
                results.append((self.chunks[idx], float(score)))
        return results

    def save(self, index_path: str = INDEX_PATH, chunks_path: str = CHUNKS_PATH) -> None:
        """Persist the FAISS index and chunk metadata to disk."""
        faiss.write_index(self.index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"[VectorStore] Saved index ({self.index.ntotal} vectors) to {index_path}")

    @classmethod
    def load(cls, index_path: str = INDEX_PATH, chunks_path: str = CHUNKS_PATH) -> "MLBVectorStore":
        """Load a previously saved vector store from disk."""
        store = cls()
        store.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            store.chunks = pickle.load(f)
        print(f"[VectorStore] Loaded {store.index.ntotal} vectors from {index_path}")
        return store

    @property
    def size(self) -> int:
        return self.index.ntotal


# ── Build Pipeline ─────────────────────────────────────────────────────────────

def build_vector_store(
    chunks: List[MLBChunk],
    embedder: MLBEmbedder = None,
    save: bool = True
) -> MLBVectorStore:
    """
    End-to-end: embed chunks → build FAISS index → optionally save.

    Args:
        chunks: List of MLBChunk objects from data_ingestion.py.
        embedder: MLBEmbedder instance. Creates one if not provided.
        save: Whether to persist the index to disk.

    Returns:
        Populated MLBVectorStore ready for querying.
    """
    if embedder is None:
        embedder = MLBEmbedder()

    print(f"[Pipeline] Embedding {len(chunks)} chunks...")
    embeddings = embedder.embed_chunks(chunks)

    store = MLBVectorStore(embedding_dim=embeddings.shape[1])
    store.add(chunks, embeddings)

    if save:
        store.save()

    return store


# ── Query Helper ───────────────────────────────────────────────────────────────

def query_store(
    query: str,
    store: MLBVectorStore,
    embedder: MLBEmbedder,
    top_k: int = 5,
    chunk_type_filter: Optional[str] = None
) -> List[Tuple[MLBChunk, float]]:
    """
    Embed a natural language query and retrieve the top-k matching chunks.

    Args:
        query: Natural language question or topic (e.g., "who hit a home run today?")
        store: Populated MLBVectorStore.
        embedder: MLBEmbedder for encoding the query.
        top_k: Number of results.
        chunk_type_filter: If set, only return chunks of this type.

    Returns:
        List of (MLBChunk, similarity_score) tuples.
    """
    query_emb = embedder.embed([query], show_progress=False)
    results = store.search(query_emb, top_k=top_k * 2)   # over-fetch for filtering

    if chunk_type_filter:
        results = [(c, s) for c, s in results if c.chunk_type == chunk_type_filter]

    return results[:top_k]


# ── Quick Test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_ingestion import ingest_mlb_data

    # 1. Ingest
    chunks = ingest_mlb_data(days_back=2)

    # 2. Embed + index
    embedder = MLBEmbedder()
    store = build_vector_store(chunks, embedder=embedder, save=True)

    # 3. Test retrieval
    test_queries = [
        "which teams won yesterday?",
        "who are the best teams in the AL East?",
        "any close games recently?",
    ]

    print("\n── Retrieval Test ──")
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        results = query_store(q, store, embedder, top_k=3)
        for chunk, score in results:
            print(f"  [{score:.3f}] ({chunk.chunk_type}) {chunk.text[:120]}...")
