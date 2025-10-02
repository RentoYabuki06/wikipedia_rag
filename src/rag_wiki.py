import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

from embedder import E5Embedder
from vector_store import FAISSVectorStore
from reranker import BGEReranker
from generator import QwenGenerator
from config import DEFAULT_GENERATOR_MODEL, DEFAULT_ARTIFACTS_DIR, DEFAULT_TOP_K, DEFAULT_TOP_N


class WikiRAG:
    def __init__(self, artifacts_dir: str = DEFAULT_ARTIFACTS_DIR, generator_model: str = DEFAULT_GENERATOR_MODEL):
        """WikiRAGシステムを初期化"""
        self.artifacts_dir = Path(artifacts_dir)
        self.generator_model = generator_model
        self.embedder = None
        self.vector_store = None
        self.reranker = None
        self.generator = None
        self.metadata = []

        self._load_components()

    def _load_components(self):
        """必要なコンポーネントを読み込み"""
        try:
            # 埋め込みモデル
            logging.info("Loading embedding model...")
            self.embedder = E5Embedder()

            # ベクトルストア
            logging.info("Loading vector store...")
            # self.vector_store = FAISSVectorStore(768)
            index_path = self.artifacts_dir / "wiki.index"
            self.vector_store = FAISSVectorStore.load(str(index_path))

            # メタデータ
            logging.info("Loading metadata...")
            self._load_metadata()

            # 再ランカー（オプション）
            logging.info("Loading reranker...")
            self.reranker = BGEReranker()

            # LLM
            logging.info(f"Loading generator ({self.generator_model})...")
            self.generator = QwenGenerator(model_name=self.generator_model)

            logging.info("All components loaded successfully")

        except Exception as e:
            logging.error(f"Failed to load components: {e}")
            raise

    def _load_metadata(self):
        """メタデータを読み込み"""
        metadata_path = self.artifacts_dir / "wiki_metas.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.metadata = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line.strip()))

        logging.info(f"Loaded {len(self.metadata)} metadata records")

    def search_and_generate(
        self, question: str, top_k: int = DEFAULT_TOP_K, top_n: int = DEFAULT_TOP_N, use_rerank: bool = True
    ) -> Dict[str, Any]:
        """質問に対する回答を生成"""
        try:
            # Step 1: 質問の埋め込み
            logging.info(f"Processing question: {question}")
            query_embedding = self.embedder.encode_query(question)

            # Step 2: ベクトル検索
            logging.info(f"Searching top-{top_k} candidates...")
            scores, indices = self.vector_store.search(query_embedding, k=top_k)

            # Step 3: 候補の取得
            candidates = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.metadata):
                    candidate = self.metadata[idx].copy()
                    candidate["vector_score"] = float(score)
                    candidate["rank"] = i
                    candidates.append(candidate)

            logging.info(f"Retrieved {len(candidates)} candidates")

            # Step 4: 再ランキング（オプション）
            final_candidates = candidates[:top_n]  # デフォルトはベクトル検索順

            if use_rerank and self.reranker.is_available:
                logging.info(f"Reranking top-{top_n} candidates...")
                passages = [c["text"] for c in candidates]
                rerank_results = self.reranker.rerank(question, passages, top_k=top_n)

                final_candidates = []
                for rank, (orig_idx, rerank_score) in enumerate(rerank_results):
                    candidate = candidates[orig_idx].copy()
                    candidate["rerank_score"] = rerank_score
                    candidate["final_rank"] = rank
                    final_candidates.append(candidate)

            # Step 5: 回答生成
            if final_candidates:
                logging.info("Generating answer...")
                answer = self.generator.generate_answer(question, final_candidates)
            else:
                answer = (
                    "該当するコンテキストが見つかりませんでした。質問を言い換えるか、より一般的な表現を試してください。"
                )

            # 結果をまとめる
            result = {
                "question": question,
                "answer": answer,
                "contexts": final_candidates,
                "search_stats": {
                    "total_candidates": len(candidates),
                    "final_candidates": len(final_candidates),
                    "rerank_used": use_rerank and self.reranker.is_available,
                },
            }

            return result

        except Exception as e:
            logging.error(f"Error during search and generation: {e}")
            return {
                "question": question,
                "answer": f"回答の生成中にエラーが発生しました: {e}",
                "contexts": [],
                "search_stats": {"error": str(e)},
            }


def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description="Wikipedia RAG質問応答")
    parser.add_argument("-q", "--question", required=True, help="質問文")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K, help="ベクトル検索で取得する候補数")
    parser.add_argument("--topn", type=int, default=DEFAULT_TOP_N, help="最終的に使用する候補数")
    parser.add_argument("--no-rerank", action="store_true", help="再ランキングを無効化")
    parser.add_argument("--artifacts_dir", default=DEFAULT_ARTIFACTS_DIR, help="アーティファクトディレクトリ")
    parser.add_argument(
        "--model",
        default=DEFAULT_GENERATOR_MODEL,
        help="生成モデル名 (例: Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument("--verbose", action="store_true", help="詳細ログを表示")
    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """ログ設定"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    args = parse_arguments()
    setup_logging(args.verbose)

    try:
        # RAGシステム初期化
        rag = WikiRAG(args.artifacts_dir, generator_model=args.model)

        # 質問処理
        result = rag.search_and_generate(
            question=args.question, top_k=args.topk, top_n=args.topn, use_rerank=not args.no_rerank
        )

        # 結果出力
        print("=" * 50)
        print(f"質問: {result['question']}")
        print("=" * 50)
        print(f"回答:\n{result['answer']}")
        print("=" * 50)

        if args.verbose and result["contexts"]:
            print("参照コンテキスト:")
            for i, ctx in enumerate(result["contexts"]):
                print(f"[{i}] {ctx.get('article_title', 'Unknown')} (スコア: {ctx.get('vector_score', 0):.3f})")
                print(f"    {ctx['text'][:100]}...")
            print("=" * 50)

        print(f"検索統計: {result['search_stats']}")

    except Exception as e:
        logging.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
