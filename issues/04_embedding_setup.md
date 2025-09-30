# Issue #04: 埋め込みモデル（E5）のセットアップと実装

**予想時間**: 30分
**難易度**: 中級
**学習項目**: 埋め込みモデル、Sentence Transformers、E5モデルの特殊な使用法

## 概要
multilingual-e5-baseモデルを使用して、テキストチャンクの埋め込みベクトルを生成する機能を実装する。

## タスク

### 1. 埋め込みクラスの実装 (`src/embedder.py`)
- `sentence-transformers`を使用したE5モデルの初期化
- E5の特殊な前置詞（"passage: ", "query: "）の正しい実装
- バッチ処理による効率的な埋め込み生成

### 2. E5モデルの特殊仕様の実装
- Passage埋め込み時: `"passage: " + text`
- Query埋め込み時: `"query: " + text`
- L2正規化の適用（内積計算でコサイン類似度と等価にするため）

### 3. バッチ処理機能
- 大量のチャンクを効率的に処理
- メモリ使用量の制御
- 進捗表示機能

### 4. 埋め込み結果の保存
- NumPy配列として保存
- チャンクIDとの対応関係の維持
- 再利用可能な形式での保存

## 実装例のスケルトン
```python
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging

class E5Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        """E5埋め込みモデルを初期化"""
        self.model = SentenceTransformer(model_name)
        logging.info(f"Loaded embedding model: {model_name}")
    
    def encode_passages(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """パッセージ（文書）の埋め込みを生成"""
        pass
    
    def encode_query(self, query: str) -> np.ndarray:
        """クエリの埋め込みを生成"""
        pass
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """L2正規化を適用"""
        pass

def save_embeddings(embeddings: np.ndarray, filepath: str) -> None:
    """埋め込みをファイルに保存"""
    pass

def load_embeddings(filepath: str) -> np.ndarray:
    """埋め込みをファイルから読み込み"""
    pass
```

## 重要なポイント

### E5モデルの前置詞について
E5モデルは以下の前置詞を**必須**で要求します：
- 文書側: `"passage: " + 元のテキスト`
- クエリ側: `"query: " + 質問文`

### 正規化について
- 埋め込みベクトルはL2正規化を適用
- これにより内積計算でコサイン類似度と等価になる

## 受け入れ基準
- [ ] E5モデルが正常に読み込まれる
- [ ] 前置詞が正しく適用される
- [ ] バッチ処理が効率的に動作する
- [ ] L2正規化が適用される
- [ ] 埋め込み結果が保存・読み込み可能
- [ ] 進捗が適切に表示される

## 学習ポイント
- 埋め込みモデルの基本原理
- E5モデルの特殊な仕様
- バッチ処理による効率化
- ベクトル正規化の重要性
- メモリ効率的なデータ処理

## テスト方法
```bash
python -c "
from src.embedder import E5Embedder
embedder = E5Embedder()
passages = ['これはテストです。', '日本語の文章です。']
embeddings = embedder.encode_passages(passages)
print(f'Generated embeddings shape: {embeddings.shape}')
query_emb = embedder.encode_query('テスト')
print(f'Query embedding shape: {query_emb.shape}')
"
```

## 次のステップ
Issue #05: FAISSインデックスの構築と保存