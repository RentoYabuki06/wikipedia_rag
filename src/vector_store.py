import faiss
import numpy as np
import logging
from typing import List, Tuple, Optional


class FAISSVectorStore:
    def __init__(self, dimension: int) -> None:
        """FAISSベクトルストアを初期化"""
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.is_trained = True  # IndexFlatIPは訓練不要

    def build_index(self, embeddings: np.ndarray) -> None:
        """埋め込みベクトルを一括登録"""
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch {embeddings.shape[1]} vs {self.dimension}")
        self.index.add(embeddings)
        logging.info(f"Added {embeddings.shape[0]} vectors to index (dim={self.dimension})")

    def add_vectors(self, vectors: np.ndarray) -> None:
        """ベクトル追加登録"""
        if vectors.shape[1] != self.dimension:
            raise ValueError("Dimension mismatch")
        self.index.add(vectors)

    def search(self, query: np.ndarray, k: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """
        類似ベクトルを検索
        Returns: (scores, indices)
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        scores, indices = self.index.search(query, k)
        return scores, indices

    def save(self, filepath: str) -> None:
        """インデックスをファイルに保存"""
        faiss.write_index(self.index, filepath)
        logging.info(f"FAISS index saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "FAISSVectorStore":
        """ファイルからインデックスを読み込み"""
        index = faiss.read_index(filepath)
        dimension = index.d
        store = cls(dimension)
        store.index = index
        return store

    def get_stats(self) -> dict:
        """インデックス統計を取得"""
        return {"ntotal": self.index.ntotal, "dimension": self.dimension, "is_trained": self.is_trained}
