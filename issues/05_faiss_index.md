# Issue #05: FAISSインデックスの構築と保存

**予想時間**: 25分
**難易度**: 中級
**学習項目**: FAISS、ベクトル検索、インデックス構築

## 概要
埋め込みベクトルをFAISSインデックスに登録し、高速な類似度検索を可能にする機能を実装する。

## タスク

### 1. FAISSインデックス管理クラスの実装 (`src/vector_store.py`)
- `IndexFlatIP`（内積検索）を使用したインデックス作成
- 埋め込みベクトルの一括登録
- インデックスの保存・読み込み機能

### 2. インデックス構築機能
- 正規化済み埋め込みベクトルの登録
- インデックスサイズとベクトル次元の検証
- 構築統計の出力（登録ベクトル数、次元数など）

### 3. 検索機能の実装
- Top-K検索の実装
- スコア（類似度）の取得
- 検索結果とチャンクIDの対応付け

### 4. ファイル入出力
- インデックスファイルの保存（`artifacts/wiki.index`）
- インデックス読み込み時の整合性チェック
- メタデータとの同期確認

## 実装例のスケルトン
```python
import faiss
import numpy as np
import logging
from typing import List, Tuple, Optional

class FAISSVectorStore:
    def __init__(self, dimension: int):
        """FAISSベクトルストアを初期化"""
        self.dimension = dimension
        self.index = None
        self.is_trained = False
    
    def build_index(self, embeddings: np.ndarray) -> None:
        """埋め込みからインデックスを構築"""
        pass
    
    def add_vectors(self, vectors: np.ndarray) -> None:
        """ベクトルをインデックスに追加"""
        pass
    
    def search(self, query_vector: np.ndarray, k: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """類似ベクトルを検索"""
        pass
    
    def save(self, filepath: str) -> None:
        """インデックスをファイルに保存"""
        pass
    
    def load(self, filepath: str) -> None:
        """ファイルからインデックスを読み込み"""
        pass
    
    def get_stats(self) -> dict:
        """インデックス統計を取得"""
        pass
```

## 重要なポイント

### IndexFlatIPについて
- 内積（Inner Product）による検索
- L2正規化されたベクトルでコサイン類似度と等価
- 小〜中規模データセットに適している

### 検索戦略
1. まずTop-16で候補を取得
2. 後続のrerankerでTop-5に絞り込み

## 受け入れ基準
- [ ] FAISSインデックスが正常に作成される
- [ ] 埋め込みベクトルが正しく登録される
- [ ] Top-K検索が期待通りに動作する
- [ ] インデックスの保存・読み込みが成功する
- [ ] ベクトル次元の検証が動作する
- [ ] 検索結果のスコアが妥当な範囲内

## 学習ポイント
- FAISSライブラリの基本的な使い方
- ベクトル検索の仕組み
- インデックスタイプの選択基準
- 内積とコサイン類似度の関係
- 大規模ベクトル検索の効率化

## テスト方法
```bash
python -c "
import numpy as np
from src.vector_store import FAISSVectorStore

# テスト用ダミーデータ
embeddings = np.random.randn(1000, 768).astype(np.float32)
# L2正規化
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

store = FAISSVectorStore(768)
store.build_index(embeddings)
print(f'Index stats: {store.get_stats()}')

# テスト検索
query = np.random.randn(1, 768).astype(np.float32)
query = query / np.linalg.norm(query)
scores, indices = store.search(query, k=5)
print(f'Search results: indices={indices[0]}, scores={scores[0]}')
"
```

## 次のステップ
Issue #06: インデックス構築用統合スクリプト（build_wiki_index.py）の実装