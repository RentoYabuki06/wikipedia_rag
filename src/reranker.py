import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Optional
import logging
import warnings

class BGEReranker:
	def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_gpu: bool = True) -> None:
		"""BGE再ランカーを初期化"""
		self.model_name = model_name
		self.device = self._setup_device(use_gpu)
		self.tokenizer = None
		self.model = None
		self.is_available = False

		try:
			self._load_model()
			self.is_available = True
			logging.info(f"Reranker loaded successfully: {model_name}")
		except Exception as e:
			logging.warning(f"Failed to load reranker: {e}")
			logging.info("Reranking will be skipped")

	def _setup_device(self, use_gpu: bool) -> torch.device:
		"""デバイスを設定"""
		if use_gpu and torch.cuda.is_available():
			return torch.device("cuda")
		else:
			return torch.device("cpu")

	def _load_model(self):
		"""モデルとトークナイザーを読み込み"""
		self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
		self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
		self.model.to(self.device)
		self.model.eval()

	def rerank(self, query: str, passages: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
		"""パッセージ再ランク"""
		if not self.is_available:
			# BGEモデルが使えない場合はダミーのスコアを返す [(0, 1.0), (1, 0.9), (2, 0.8), ...]
			return [(i, 1.0 - i*0.1) for i in range(min(top_k, len(passages)))]
		try:
			return self._compute_rerank_scores(query, passages, top_k)
		except Exception as e:
			logging.warning(f"Reranking failed, using original order: {e}")
			return [(i, 1.0 - i*0.1) for i in range(min(top_k, len(passages)))]
		
	def _compute_rerank_scores(self, query: str, passages: List[str], top_k: int) -> List[Tuple[int, float]]:
		"""実際の再ランクスコア計算"""
		pairs = self._create_pairs(query, passages)
		inputs = self.tokenizer(
			pairs,
			padding=True,		# バッチ内で最長の入力に合わせて自動でパディング（埋め草）する
			truncation=True,	# max_lengthを超えた場合、自動で切り詰める
			max_length=512,		# モデルが受け付ける最大トークン長
			return_tensors="pt"	# 結果をPyTorchのテンソル torch.Tensorとして返す
		)
		# tokenizerが返した各テンソルを指定したデバイスに転送し、それを新しい辞書として再構築
		inputs = {k: v.to(self.device) for k, v in inputs.items()}
		# Pytorchで勾配計算を無効化
		with torch.no_grad():
			outputs = self.model(**inputs)
			# BGEはラベル0のスコアが関連度
			if hasattr(outputs, "logits"):
				scores = outputs.logits.squeeze(-1).cpu().numpy()
			else:
				scores = outputs[0].squeeze(-1).cpu().numpy()
		# スコアで降順ソートし、上位 top_k のインデックスとスコアを返す
		ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
		return [(idx, float(score)) for idx, score in ranked[:top_k]]

	def _create_pairs(self, query: str, passages: List[str]) -> List[str]:
		"""クエリ-パッセージペアを作成"""
		# BGEは [query] [SEP] [passage] 形式
		return [f"{query} [SEP] {passage}" for passage in passages]

def combine_scores(vector_scores: List[float], rerank_scores: List[float], alpha: float = 0.7) -> List[float]:
	"""ベクトル検索スコアと再ランクスコアを統合"""
	# alpha: ベクトル検索スコアの重み, (1-alpha): 再ランクスコアの重み
	return [alpha * v + (1 - alpha) * r for v, r in zip(vector_scores, rerank_scores)]