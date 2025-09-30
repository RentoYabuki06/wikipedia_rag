# Issue #03: テキスト分割（チャンキング）機能の実装

**予想時間**: 25分
**難易度**: 中級
**学習項目**: 自然言語処理、文字ベース分割、オーバーラップ処理

## 概要
Wikipediaの長い記事を適切なサイズのチャンクに分割し、検索効率を向上させる機能を実装する。

## タスク

### 1. チャンキング機能の実装 (`src/chunker.py`)
- 文字ベースでの分割（デフォルト: size=450, overlap=60）
- オーバーラップ機能で文脈の連続性を保持
- 日本語テキストに適した分割ポイントの調整

### 2. チャンクメタデータの生成
各チャンクに以下の情報を付与：
```python
{
    "id": str,              # chunk_<article_id>_<chunk_num>
    "source": str,          # jawiki:<title>
    "chunk_id": int,        # チャンク番号（0から開始）
    "text": str,            # チャンクテキスト
    "article_title": str,   # 元記事タイトル
    "start_char": int,      # 元記事での開始文字位置
    "end_char": int         # 元記事での終了文字位置
}
```

### 3. 品質制御
- 最小チャンクサイズの設定（短すぎるチャンクの除外）
- 空白のみのチャンクの除外
- 適切な境界での分割（句読点での区切り優先など）

### 4. 出力機能
- チャンクデータをJSONL形式で保存（`artifacts/wiki_metas.jsonl`）
- 処理統計の出力（総チャンク数、平均長さなど）

## 実装例のスケルトン
```python
import json
from typing import List, Dict, Any

class TextChunker:
    def __init__(self, chunk_size: int = 450, overlap: int = 60):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, article_id: str, title: str) -> List[Dict[str, Any]]:
        """テキストをチャンクに分割"""
        pass
    
    def find_split_point(self, text: str, target_pos: int) -> int:
        """適切な分割点を探す"""
        pass

def save_chunks_to_jsonl(chunks: List[Dict], filepath: str) -> None:
    """チャンクをJSONLファイルに保存"""
    pass
```

## 受け入れ基準
- [ ] 指定サイズでテキストが分割される
- [ ] オーバーラップが正しく機能する
- [ ] チャンクメタデータが正確に生成される
- [ ] JSONL形式で正しく保存される
- [ ] 日本語テキストで適切な分割が行われる
- [ ] 短すぎるチャンクが適切に除外される

## 学習ポイント
- テキスト分割アルゴリズムの理解
- RAGにおけるチャンクサイズの重要性
- JSONLフォーマットの扱い方
- 日本語テキスト処理の特殊性

## テスト方法
```bash
python -c "
from src.data_loader import load_wikipedia_data
from src.chunker import TextChunker
data = load_wikipedia_data(10)
chunker = TextChunker()
chunks = []
for article in data:
    chunks.extend(chunker.chunk_text(article['text'], article['id'], article['title']))
print(f'Generated {len(chunks)} chunks from {len(data)} articles')
"
```

## 次のステップ
Issue #04: 埋め込みモデル（E5）のセットアップと実装