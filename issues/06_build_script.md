# Issue #06: インデックス構築用統合スクリプト（build_wiki_index.py）の実装

**予想時間**: 30分
**難易度**: 中級
**学習項目**: パイプライン設計、CLI引数処理、エラーハンドリング

## 概要
これまで実装した機能を統合し、ワンコマンドでWikipediaデータからFAISSインデックスまでを構築するスクリプトを作成する。

## タスク

### 1. メインスクリプトの作成 (`src/build_wiki_index.py`)
- データ取得からインデックス構築までの完全パイプライン
- コマンドライン引数による設定変更機能
- 詳細なログ出力とプログレス表示

### 2. パイプライン実行順序
1. Wikipediaデータ取得・正規化
2. テキストのチャンク分割
3. チャンクメタデータの保存
4. 埋め込み生成
5. FAISSインデックス構築・保存

### 3. CLI引数の実装
```bash
python src/build_wiki_index.py \
  --max_articles 30000 \
  --config "20231101.ja" \
  --chunk_size 450 \
  --overlap 60 \
  --batch_size 32 \
  --output_dir "artifacts"
```

### 4. エラーハンドリングと復旧機能
- 各ステップでの例外処理
- 中間ファイルの存在チェック（再開機能）
- メモリ不足時の対応

## 実装例のスケルトン
```python
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
    pass

def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description="Wikipedia RAGインデックス構築")
    parser.add_argument("--max_articles", type=int, default=30000)
    parser.add_argument("--config", default="20231101.ja")
    parser.add_argument("--chunk_size", type=int, default=450)
    parser.add_argument("--overlap", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="artifacts")
    return parser.parse_args()

def build_wiki_index(args):
    """メインの構築処理"""
    try:
        # Step 1: データ取得
        logging.info("Starting Wikipedia data loading...")
        
        # Step 2: チャンク分割
        logging.info("Starting text chunking...")
        
        # Step 3: 埋め込み生成
        logging.info("Starting embedding generation...")
        
        # Step 4: インデックス構築
        logging.info("Building FAISS index...")
        
        logging.info("Index building completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during index building: {e}")
        sys.exit(1)

def main():
    args = parse_arguments()
    setup_logging()
    
    # 出力ディレクトリ作成
    Path(args.output_dir).mkdir(exist_ok=True)
    
    build_wiki_index(args)

if __name__ == "__main__":
    main()
```

## 進捗表示の実装
- 各ステップの開始・完了時刻
- 処理件数とETA（予想残り時間）
- メモリ使用量の監視

## 出力ファイル
実行完了後、以下のファイルが生成される：
- `artifacts/wiki_metas.jsonl` - チャンクメタデータ
- `artifacts/wiki.index` - FAISSインデックス
- `artifacts/wiki_embeddings.npy` - 埋め込みベクトル（オプション）

## 受け入れ基準
- [ ] 全ステップが順次正常に実行される
- [ ] CLI引数が適切に処理される
- [ ] エラー時に適切なメッセージが出力される
- [ ] 進捗が視覚的に分かりやすい
- [ ] 出力ファイルが期待される場所に生成される
- [ ] 実行統計が最後に表示される

## 学習ポイント
- パイプライン設計のベストプラクティス
- argparseを使ったCLI作成
- Python loggingの実践的な使い方
- 大規模データ処理時のメモリ管理
- エラーハンドリング戦略

## テスト方法
```bash
# 小さなデータセットでのテスト
python src/build_wiki_index.py --max_articles 100 --batch_size 16

# 設定を変更してのテスト
python src/build_wiki_index.py --max_articles 1000 --chunk_size 300 --overlap 30
```

## 実行例出力
```
2025-09-30 10:00:00 - INFO - Starting Wikipedia data loading...
2025-09-30 10:00:05 - INFO - Loaded 30,000 articles
2025-09-30 10:00:05 - INFO - Starting text chunking...
2025-09-30 10:00:15 - INFO - Generated 95,234 chunks
2025-09-30 10:00:15 - INFO - Starting embedding generation...
2025-09-30 10:05:20 - INFO - Generated embeddings for 95,234 chunks
2025-09-30 10:05:20 - INFO - Building FAISS index...
2025-09-30 10:05:25 - INFO - FAISS index created: 95,234 vectors, 768 dimensions
2025-09-30 10:05:25 - INFO - Index building completed successfully!
```

## 次のステップ
Issue #07: 再ランキング機能の実装（BGE Reranker）