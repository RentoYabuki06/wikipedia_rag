import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging

class E5Embedder:
	def __init__(self, model_name: str = "intfloat/multilingual-e5-base") -> None:
		"""E5埋め込みモデルの初期化"""
		self.model = SentenceTransformer(model_name)
		logging.info(f"Loaded embedding model: {model_name}")
		
	def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
		"""L2正則化を適用"""
		norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
		return embeddings / (norms + 1e-10)

	def encode_passages(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
		"""パッセージ（文章）の埋め込みを生成"""
		passages = ["passage: " + t for t in texts]
		embeddings = self.model.encode(passages, batch_size=batch_size, show_progress_bar=True)
		return self._normalize_embeddings(np.array(embeddings))

	def encode_query(self, query: str) -> np.ndarray:
		""""クエリの埋め込みを生成"""
		q = "query: " + query
		emb = self.model.encode([q])
		return self._normalize_embeddings(np.array(emb))[0]
	
	
def save_embeddings(embeddings: np.ndarray, filepath: str) -> None:
	"""埋め込みをファイルに保存"""
	np.save(filepath, embeddings)

def load_embeddings(filepath: str) -> np.ndarray:
	"""埋め込みをファイルから読み込み"""
	return np.load(filepath)