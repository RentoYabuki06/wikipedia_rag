import logging
from datasets import load_dataset
from typing import List, Dict, Any
import re

def load_wikipedia_data(max_articles: int = 30000, config: str = "20231101.ja") -> List[Dict[str, Any]]:
	logging.basicConfig(level=logging.INFO)
	try:
		dataset = load_dataset("wikimedia/wikipedia", config, split="train")
		logging.info(f"Loaded Wikipedia dataset: {config}")
	except Exception as e:
		logging.error(f"Failed to load dataset: {e}")
		return []
	
	articles = []
	count = 0
	for item in dataset:
		if count >= max_articles:
			break
		# keyを取得（keyが存在しない or 値がnoneの場合はデフォルトの空文字列が返ってくる）
		text = item.get("text", "")
		title = item.get("title", "")
		if not text or not title:
			continue
		norm_text = normalize_text(text)
		if not norm_text.strip():
			continue
		article = {
			"id": str(item.get("id", count)),
			"title": title,
			"text": norm_text,
			"source": f"jawiki:{title}"
		}
		articles.append(article)
		count += 1
		if count % 1000 == 0:
			logging.info(f"Processed {count} articles")
	logging.info(f"Total articles loaded: {len(articles)}")
	return articles

def normalize_text(text: str) -> str:
	# 連続する空白文字を半角スペースに変換
	text = re.sub(r'\s+', ' ', text)
	# 前後の空白文字を取り除く
	text = text.strip()
	# 改行文字の正規化（windowsの改行CR+LFをLFへ、古いMacの改行CRをLFへ）
	text = text.replace('\r\n', '\n').replace('\r', '\n')
	return text
