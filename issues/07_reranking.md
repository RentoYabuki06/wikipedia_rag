# Issue #07: 再ランキング機能の実装（BGE Reranker）

**予想時間**: 25分
**難易度**: 中級
**学習項目**: Cross-encoder、再ランキング、検索精度向上

## 概要
FAISS検索で得られた候補を、より精密なcross-encoderモデル（BGE Reranker）で再評価・再順序付けする機能を実装する。

## タスク

### 1. 再ランカークラスの実装 (`src/reranker.py`)
- `BAAI/bge-reranker-v2-m3`モデルの初期化
- クエリと候補文書ペアのスコアリング
- バッチ処理による効率化
- GPUが利用できない環境での適切なフォールバック

### 2. スコアリング機能
- クエリと複数候補のペアワイズスコア計算
- スコアによる候補の再順序付け
- 元の検索スコアとの統合オプション

### 3. エラーハンドリング
- モデル読み込み失敗時のフォールバック処理
- メモリ不足時の対応
- GPU/CPU自動切り替え

### 4. パフォーマンス最適化
- バッチサイズの動的調整
- 不要な再計算の回避
- キャッシュ機能（オプション）

## 実装例のスケルトン
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Optional
import logging
import warnings

class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_gpu: bool = True):
        """BGE再ランカーを初期化"""
        self.model_name = model_name
        self.device = self._setup_device(use_gpu)
        self.tokenizer = None
        self.model = None
        self.is_available = False
        
        try:
            self._load_model()
            self.is_available = True
            logging.info(f"Reranker loaded successfully: {model_name}")
        except Exception as e:
            logging.warning(f"Failed to load reranker: {e}")
            logging.info("Reranking will be skipped")
    
    def _setup_device(self, use_gpu: bool) -> torch.device:
        """デバイスを設定"""
        pass
    
    def _load_model(self):
        """モデルとトークナイザーを読み込み"""
        pass
    
    def rerank(self, query: str, passages: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """パッセージを再ランク"""
        if not self.is_available:
            # フォールバック: 元の順序を返す
            return [(i, 1.0 - i*0.1) for i in range(min(top_k, len(passages)))]
        
        try:
            return self._compute_rerank_scores(query, passages, top_k)
        except Exception as e:
            logging.warning(f"Reranking failed, using original order: {e}")
            return [(i, 1.0 - i*0.1) for i in range(min(top_k, len(passages)))]
    
    def _compute_rerank_scores(self, query: str, passages: List[str], top_k: int) -> List[Tuple[int, float]]:
        """実際の再ランクスコア計算"""
        pass
    
    def _create_pairs(self, query: str, passages: List[str]) -> List[str]:
        """クエリ-パッセージペアを作成"""
        pass

# ユーティリティ関数
def combine_scores(vector_scores: List[float], rerank_scores: List[float], 
                  alpha: float = 0.7) -> List[float]:
    """ベクトル検索スコアと再ランクスコアを統合"""
    pass
```

## 重要なポイント

### Cross-encoderの特徴
- より精密だが計算量が大きい
- クエリと候補のペアごとに評価
- Bi-encoderより高精度な関連度評価

### フォールバック戦略
- モデル読み込み失敗→元の順序を維持
- GPU不足→CPU使用またはスキップ
- 実行時エラー→警告を出して処理続行

## 受け入れ基準
- [ ] BGEモデルが正常に読み込まれる
- [ ] クエリ-文書ペアのスコアが適切に計算される
- [ ] スコアに基づく再順序付けが機能する
- [ ] モデル利用不可時の適切なフォールバック
- [ ] GPU/CPU環境の自動判定
- [ ] バッチ処理による効率化

## 学習ポイント
- Cross-encoderとBi-encoderの違い
- 再ランキングによる検索精度向上の仕組み
- PyTorchモデルのデバイス管理
- エラーハンドリングとフォールバック設計
- 計算資源に応じた処理の調整

## テスト方法
```bash
python -c "
from src.reranker import BGEReranker

reranker = BGEReranker()
query = '日本の歴史について'
passages = [
    '日本の歴史は古代から現代まで続いています。',
    '寿司は日本料理の代表的な食べ物です。',
    '江戸時代は徳川幕府が統治していました。'
]

if reranker.is_available:
    results = reranker.rerank(query, passages, top_k=2)
    print(f'Reranking results: {results}')
else:
    print('Reranker not available, using fallback')
"
```

## 設定オプション
```python
# 高精度モード（GPU推奨）
reranker = BGEReranker(use_gpu=True)

# 軽量モード（CPU使用）
reranker = BGEReranker(use_gpu=False)

# 無効化（デバッグ用）
reranker = BGEReranker(model_name=None)
```

## 次のステップ
Issue #08: LLM応答生成機能の実装（Qwen2.5）