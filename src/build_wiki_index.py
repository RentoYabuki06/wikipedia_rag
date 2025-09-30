import argparse
import logging
import os
from pathlib import Path
import sys

from data_loader import load_wikipedia_data
from chunker import TextChunker, save_chunks_to_jsonl
from embedder import E5Embedder, save_embeddings
from vector_store import FAISSVectorStore


def setup_logging(level=logging.INFO):
	"""ログ設定を初期化"""
	logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def parse_arguments():
	"""コマンドライン引数を解析（短縮版も対応）"""
	parser = argparse.ArgumentParser(description="Wikipedia RAGインデックス構築")
	parser.add_argument("-m", "--max_articles", type=int, default=3000, help="最大記事数")
	parser.add_argument("-c", "--config", default="20231101.ja", help="Wikipediaデータセットのconfig名")
	parser.add_argument("-s", "--chunk_size", type=int, default=450, help="チャンクサイズ")
	parser.add_argument("-o", "--overlap", type=int, default=60, help="チャンクのオーバーラップ")
	parser.add_argument("-b", "--batch_size", type=int, default=32, help="埋め込み生成時のバッチサイズ")
	parser.add_argument("-d", "--output_dir", default="artifacts", help="出力ディレクトリ")
	return parser.parse_args()


def build_wiki_index(args):
	"""メインの構築処理"""
	try:
		# Step 1: データ取得
		logging.info("Starting Wikipedia data loading...")
		articles = load_wikipedia_data(max_articles=args.max_articles, config=args.config)
		logging.info(f"Loaded {len(articles)} articles")

		# Step 2: チャンク分解
		logging.info("Starting text chunking")
		chunker = TextChunker(chunk_size=args.chunk_size, overlap=args.overlap)
		all_chunks = []
		for art in articles:
			all_chunks.extend(chunker.chunk_text(art["text"], art["id"], art["title"]))
		logging.info(f"Generated {len(all_chunks)} chunks")
		metas_path = f"{args.output_dir}/wiki_metas.jsonl"
		save_chunks_to_jsonl(all_chunks, metas_path)
		logging.info(f"Saved chunk metadata to {metas_path}")

		# Step 3: 埋め込み生成
		logging.info("Starting embedding generation...")
		embedder = E5Embedder()
		texts = [chunk["text"] for chunk in all_chunks]
		embeddings = embedder.encode_passages(texts, batch_size=args.batch_size)
		emb_path = f"{args.output_dir}/wiki_embeddings.npy"
		save_embeddings(embeddings, emb_path)
		logging.info(f"Saved embeddings to {emb_path}")

		# Step 4: インデックス構築
		logging.info("Building FAISS index...")
		dimension = embeddings.shape[1]
		store = FAISSVectorStore(dimension)
		store.build_index(embeddings)
		index_path = f"{args.output_dir}/wiki.index"
		store.save(index_path)
		stats = store.get_stats()
		logging.info(f"FAISS index created: {stats['ntotal']} vectors, {stats['dimension']}")
		logging.info("Index building completed successfully!")

	except Exception as e:
		logging.error(f"Error during index building: {e}")
		sys.exit(1)

def main():
	args = parse_arguments()
	setup_logging()
	Path(args.output_dir).mkdir(exist_ok=True)
	build_wiki_index(args)

if __name__ == "__main__":
	main()
