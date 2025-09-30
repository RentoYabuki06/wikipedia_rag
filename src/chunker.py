import os
import json
from typing import List, Dict, Any, Optional

class TextChunker:
	def __init__(self, chunk_size: int = 450, overlap: int = 60, min_chunk_size: int = 100) -> None:
		"""
		chunk_size: 目標チャンク長（文字数）
		overlap: チャンク間の重複文字数
		min_chunk_size: これより短いチャンクは除外
		"""
		self.chunk_size = chunk_size
		self.overlap = overlap
		self.min_chunk_size = min_chunk_size
		# 日本語で優先される区切り候補（target内のより右側にあるものを優先）
		self.preferred_delims = ["。", "．", "、", "，", "！", "？", "\n", " "]

	def chunk_text(self, text: str, article_id: str, title: str) ->List[Dict[str, Any]]:
		"""
		テキストをチャンクに分割してメタデータを返す関数
		各チャンクは以下 dict 仕様に従う
		{
			"id": str,              # chunk_<article_id>_<chunk_num>
			"source": str,          # jawiki:<title>
			"chunk_id": int,        # チャンク番号（0から開始）
			"text": str,            # チャンクテキスト
			"article_title": str,   # 元記事タイトル
			"start_char": int,      # 元記事での開始文字位置
			"end_char": int         # 元記事での終了文字位置
		}
		"""
		chunks: List[Dict[str, Any]] = []
		if not text:
			return chunks
		length = len(text)
		start = 0
		chunk_idx = 0

		while start < length:
			target = start + self.chunk_size
			if target >= length:
				end = length
			else:
				split = self.find_split_point(text, target)
				end = split if split > start else target
			chunk_text = text[start:end].strip()
			if chunk_text and len(chunk_text) >= self.min_chunk_size:
				chunk = {
					"id": f"chunk_{article_id}_{chunk_idx}",
					"source": f"jawiki{title}",
					"text": chunk_text,
					"article_title": title,
					"start_char": start,
					"end_char": end
				}
				chunks.append(chunk)
				chunk_idx += 1

			# オーバーラップを考慮して次のstart位置を決定
			next_start = end - self.overlap
			if next_start <= start:
				next_start = end
			start = next_start
		return chunks

	def find_split_point(self, text: str, target_pos: int, window:int = 100) -> int:
		"""
		target_pos の手前で適切な分割店を探す
		- 句点や改行を優先して分割位置を探す（可能な限りtarget_posに近い位置で）
		- 見つからなければtarget_posを返す
		"""
		start_search = max(0, target_pos - window)
		segment = text[start_search:target_pos]
		for delim in self.preferred_delims:
			idx = segment.rfind(delim)
			if idx != -1:
				split_pos = start_search + idx + len(delim)
				return split_pos
		return target_pos
	

def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], filepath: str) -> None:
	"""
	chunksをJSONL形式で保存。ディレクトリがなければ作成
	"""
	dir_path = os.path.dirname(filepath)
	# ディレクトリが存在しない場合は作成
	if dir_path and not os.path.exists(dir_path):
		os.makedirs(dir_path, exist_ok=True)
	
	# withを用いてリソースの確実な解放を前提
	with open(filepath, "w", encoding="utf-8") as f:	# fはただopenが返すファイルオブジェクトに名前をつけているだけ
		for c in chunks:
			f.write(json.dumps(c, ensure_ascii=False) + "\n")	# ensure_ascii=Falseで日本語がエスケープになるのを防ぐ