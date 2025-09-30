# Issue #02: Wikipediaデータ取得機能の実装

**予想時間**: 20分
**難易度**: 初級
**学習項目**: Hugging Face Datasets、データ前処理

## 概要
日本語Wikipediaデータセットを取得し、基本的な前処理を行う機能を実装する。

## タスク

### 1. データ取得スクリプトの作成 (`src/data_loader.py`)
- `datasets.load_dataset("wikipedia", "20231101.ja")`を使用
- MAX_ARTICLES（デフォルト: 30000）で記事数を制限
- 基本的なデータ検証（空記事の除外など）

### 2. データ正規化機能の実装
- 連続空白を単一スペースに変換
- 前後空白の除去
- 改行文字の正規化

### 3. メタデータ構造の定義
各記事に以下の情報を付与：
```python
{
    "id": str,           # 一意識別子
    "title": str,        # 記事タイトル  
    "text": str,         # 正規化済みテキスト
    "source": str        # "jawiki:<title>"形式
}
```

### 4. エラーハンドリング
- データセット取得失敗時の処理
- 不正なデータ形式の処理
- ログ出力の実装

## 実装例のスケルトン
```python
import logging
from datasets import load_dataset
from typing import List, Dict, Any

def load_wikipedia_data(max_articles: int = 30000, config: str = "20231101.ja") -> List[Dict[str, Any]]:
    """日本語Wikipediaデータを取得・正規化"""
    pass

def normalize_text(text: str) -> str:
    """テキストの正規化"""
    pass
```

## 受け入れ基準
- [ ] Wikipediaデータセットが正常に取得できる
- [ ] 指定した記事数で制限される
- [ ] テキストが適切に正規化される
- [ ] メタデータが正しい形式で生成される
- [ ] エラーハンドリングが適切に動作する
- [ ] ログが適切に出力される

## 学習ポイント
- Hugging Face Datasetsライブラリの使い方
- 大量データの効率的な処理方法
- データ前処理のベストプラクティス
- Python loggingの基本

## テスト方法
```bash
python -c "from src.data_loader import load_wikipedia_data; data = load_wikipedia_data(100); print(f'Loaded {len(data)} articles')"
```

## 次のステップ
Issue #03: テキスト分割（チャンキング）機能の実装