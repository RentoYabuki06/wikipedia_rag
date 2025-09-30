# Issue #09: 検索・生成統合スクリプト（rag_wiki.py）の実装

**予想時間**: 25分
**難易度**: 中級
**学習項目**: パイプライン統合、CLI設計、エンドツーエンド処理

## 概要
構築済みのインデックスを使用して、質問から回答生成までの完全なRAGパイプラインを実行するスクリプトを実装する。

## タスク

### 1. メインRAGスクリプトの実装 (`src/rag_wiki.py`)
- インデックス・メタデータの読み込み
- 質問→検索→再ランク→生成の完全パイプライン
- CLI引数による動作制御
- 詳細な処理ログ出力

### 2. RAGパイプライン処理順序
1. 構築済みインデックス・メタデータの読み込み
2. 質問の埋め込みベクトル生成
3. FAISS検索でTop-K候補取得
4. 再ランキング（オプション）でTop-N選出
5. LLMによる回答生成
6. 結果の整形・出力

### 3. CLI引数の実装
```bash
python src/rag_wiki.py \
  --question "大政奉還とは何ですか？" \
  --topk 16 \
  --topn 5 \
  --rerank \
  --artifacts_dir "artifacts"
```

### 4. エラーハンドリング
- インデックスファイル不在時の対応
- モデル読み込み失敗時の処理
- 検索結果0件時の適切なレスポンス

## 実装例のスケルトン
```python
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

class WikiRAG:
    def __init__(self, artifacts_dir: str = "artifacts"):
        """WikiRAGシステムを初期化"""
        self.artifacts_dir = Path(artifacts_dir)
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
            self.vector_store = FAISSVectorStore(768)
            index_path = self.artifacts_dir / "wiki.index"
            self.vector_store.load(str(index_path))
            
            # メタデータ
            logging.info("Loading metadata...")
            self._load_metadata()
            
            # 再ランカー（オプション）
            logging.info("Loading reranker...")
            self.reranker = BGEReranker()
            
            # LLM
            logging.info("Loading generator...")
            self.generator = QwenGenerator()
            
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
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.metadata.append(json.loads(line.strip()))
        
        logging.info(f"Loaded {len(self.metadata)} metadata records")
    
    def search_and_generate(self, question: str, top_k: int = 16, top_n: int = 5, 
                          use_rerank: bool = True) -> Dict[str, Any]:
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
                    candidate['vector_score'] = float(score)
                    candidate['rank'] = i
                    candidates.append(candidate)
            
            logging.info(f"Retrieved {len(candidates)} candidates")
            
            # Step 4: 再ランキング（オプション）
            final_candidates = candidates[:top_n]  # デフォルトはベクトル検索順
            
            if use_rerank and self.reranker.is_available:
                logging.info(f"Reranking top-{top_n} candidates...")
                passages = [c['text'] for c in candidates]
                rerank_results = self.reranker.rerank(question, passages, top_k=top_n)
                
                final_candidates = []
                for rank, (orig_idx, rerank_score) in enumerate(rerank_results):
                    candidate = candidates[orig_idx].copy()
                    candidate['rerank_score'] = rerank_score
                    candidate['final_rank'] = rank
                    final_candidates.append(candidate)
            
            # Step 5: 回答生成
            if final_candidates:
                logging.info("Generating answer...")
                answer = self.generator.generate_answer(question, final_candidates)
            else:
                answer = "該当するコンテキストが見つかりませんでした。質問を言い換えるか、より一般的な表現を試してください。"
            
            # 結果をまとめる
            result = {
                'question': question,
                'answer': answer,
                'contexts': final_candidates,
                'search_stats': {
                    'total_candidates': len(candidates),
                    'final_candidates': len(final_candidates),
                    'rerank_used': use_rerank and self.reranker.is_available
                }
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error during search and generation: {e}")
            return {
                'question': question,
                'answer': f"回答の生成中にエラーが発生しました: {e}",
                'contexts': [],
                'search_stats': {'error': str(e)}
            }

def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description="Wikipedia RAG質問応答")
    parser.add_argument("-q", "--question", required=True, help="質問文")
    parser.add_argument("--topk", type=int, default=16, help="ベクトル検索で取得する候補数")
    parser.add_argument("--topn", type=int, default=5, help="最終的に使用する候補数")
    parser.add_argument("--no-rerank", action="store_true", help="再ランキングを無効化")
    parser.add_argument("--artifacts_dir", default="artifacts", help="アーティファクトディレクトリ")
    parser.add_argument("--verbose", action="store_true", help="詳細ログを表示")
    return parser.parse_args()

def setup_logging(verbose: bool = False):
    """ログ設定"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    args = parse_arguments()
    setup_logging(args.verbose)
    
    try:
        # RAGシステム初期化
        rag = WikiRAG(args.artifacts_dir)
        
        # 質問処理
        result = rag.search_and_generate(
            question=args.question,
            top_k=args.topk,
            top_n=args.topn,
            use_rerank=not args.no_rerank
        )
        
        # 結果出力
        print("=" * 50)
        print(f"質問: {result['question']}")
        print("=" * 50)
        print(f"回答:\n{result['answer']}")
        print("=" * 50)
        
        if args.verbose and result['contexts']:
            print("参照コンテキスト:")
            for i, ctx in enumerate(result['contexts']):
                print(f"[{i}] {ctx.get('article_title', 'Unknown')} (スコア: {ctx.get('vector_score', 0):.3f})")
                print(f"    {ctx['text'][:100]}...")
            print("=" * 50)
        
        print(f"検索統計: {result['search_stats']}")
        
    except Exception as e:
        logging.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## 受け入れ基準
- [ ] 構築済みインデックスが正常に読み込まれる
- [ ] 質問から回答までのパイプラインが完動する
- [ ] CLI引数が適切に処理される
- [ ] 検索結果0件時の適切なメッセージ表示
- [ ] 詳細ログオプションが機能する
- [ ] 再ランキングの有効/無効切り替えが動作する

## 学習ポイント
- エンドツーエンドRAGパイプラインの設計
- コンポーネント間の適切なデータ受け渡し
- CLI設計のベストプラクティス
- エラーハンドリングとユーザビリティ
- デバッグ情報の効果的な出力

## テスト方法
```bash
# 基本的な質問
python src/rag_wiki.py -q "織田信長について教えて"

# 詳細ログ付き
python src/rag_wiki.py -q "江戸幕府の成立について" --verbose

# 再ランキング無効
python src/rag_wiki.py -q "明治維新とは" --no-rerank

# パラメータ調整
python src/rag_wiki.py -q "戦国時代の特徴" --topk 20 --topn 3
```

## 実行例出力
```
==================================================
質問: 大政奉還について教えてください
==================================================
回答: 大政奉還は、1867年（慶応3年）10月14日に江戸幕府第15代将軍徳川慶喜が政権を朝廷に返上した政治的事件です。これにより江戸時代が終わり、明治維新へと繋がりました。

参照: [0] jawiki:大政奉還#chunk=1, [1] jawiki:徳川慶喜#chunk=5
==================================================
検索統計: {'total_candidates': 16, 'final_candidates': 5, 'rerank_used': True}
```

## 次のステップ
Issue #10: Gradio UIの実装（app_wiki.py）