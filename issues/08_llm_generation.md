# Issue #08: LLM応答生成機能の実装（Qwen2.5）

**予想時間**: 30分
**難易度**: 上級
**学習項目**: LLM推論、プロンプトエンジニアリング、transformersライブラリ

## 概要
検索・再ランクされたコンテキストを用いて、Qwen2.5-3B-Instructで日本語の回答を生成する機能を実装する。

## タスク

### 1. LLM生成クラスの実装 (`src/generator.py`)
- Qwen2.5-3B-Instructモデルの初期化
- 適切なプロンプトテンプレートの実装
- 参照情報付きの回答生成
- GPU/CPUの自動選択

### 2. プロンプト設計
```python
SYSTEM_PROMPT = """あなたは事実に忠実なアシスタントです。与えられたコンテキストの情報のみを使用して回答してください。
- コンテキスト以外の情報は推測しないでください
- 不明な場合は「わからない」と答えてください  
- 回答の最後に参照した情報源を[0], [1]の形式で示してください"""

USER_TEMPLATE = """質問: {question}

参考コンテキスト:
{context}

上記のコンテキストのみを使用して回答してください。"""
```

### 3. 参照情報の処理
- 検索結果をコンテキストとして整形
- 参照番号とソース情報の対応付け
- 回答末尾への参照情報追加

### 4. 生成パラメータの最適化
- temperature, top_p, max_lengthの調整
- 日本語生成に適した設定
- 反復・幻覚の抑制

## 実装例のスケルトン
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List, Dict, Any, Optional
import logging

class QwenGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", use_gpu: bool = True):
        """Qwen生成モデルを初期化"""
        self.model_name = model_name
        self.device = self._setup_device(use_gpu)
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        
        try:
            self._load_model()
            logging.info(f"Generator loaded successfully: {model_name}")
        except Exception as e:
            logging.error(f"Failed to load generator: {e}")
            raise
    
    def _setup_device(self, use_gpu: bool) -> torch.device:
        """デバイスを設定"""
        pass
    
    def _load_model(self):
        """モデルとトークナイザーを読み込み"""
        pass
    
    def _setup_generation_config(self):
        """生成設定を初期化"""
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    def generate_answer(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        """質問とコンテキストから回答を生成"""
        try:
            # プロンプト構築
            prompt = self._build_prompt(question, contexts)
            
            # 生成実行
            answer = self._generate_text(prompt)
            
            # 参照情報を追加
            answer_with_sources = self._add_source_references(answer, contexts)
            
            return answer_with_sources
            
        except Exception as e:
            logging.error(f"Answer generation failed: {e}")
            return "申し訳ありませんが、回答の生成中にエラーが発生しました。"
    
    def _build_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        """プロンプトを構築"""
        pass
    
    def _generate_text(self, prompt: str) -> str:
        """テキスト生成を実行"""
        pass
    
    def _add_source_references(self, answer: str, contexts: List[Dict[str, Any]]) -> str:
        """回答に参照情報を追加"""
        pass

# ユーティリティ関数
def format_contexts(contexts: List[Dict[str, Any]]) -> str:
    """コンテキストを表示用に整形"""
    formatted = []
    for i, ctx in enumerate(contexts):
        formatted.append(f"[{i}] {ctx['text']}")
    return "\n\n".join(formatted)

def extract_source_info(context: Dict[str, Any]) -> str:
    """ソース情報を抽出"""
    return f"jawiki:{context.get('article_title', 'Unknown')}#chunk={context.get('chunk_id', 0)}"
```

## プロンプト設計のポイント

### System Prompt
- 事実重視・推測禁止を明確に指示
- 参照情報の必須化
- 不確実性の適切な表明

### Context Formatting
- 各コンテキストに番号を付与
- 読みやすい形式で区切り
- 長すぎる場合の切り詰め処理

## 生成品質の制御

### パラメータ調整
- `temperature=0.3`: 創造性を抑制、事実重視
- `top_p=0.9`: 多様性を保ちつつ安定性確保
- `repetition_penalty=1.1`: 反復を軽度に抑制

### 幻覚対策
- System promptでの明確な制約
- コンテキスト外情報の使用禁止
- 不明時の素直な表明を促進

## 受け入れ基準
- [ ] Qwenモデルが正常に読み込まれる
- [ ] 適切なプロンプト形式で入力される
- [ ] コンテキストに基づいた回答が生成される
- [ ] 参照情報が正しく付与される
- [ ] GPU/CPU環境で適切に動作する
- [ ] 日本語での自然な応答

## 学習ポイント
- 大規模言語モデルの推論プロセス
- RAGにおけるプロンプトエンジニアリング
- 生成パラメータによる出力制御
- 幻覚（Hallucination）対策
- GPU/CPUでの効率的なモデル実行

## テスト方法
```bash
python -c "
from src.generator import QwenGenerator

generator = QwenGenerator()
question = '織田信長について教えてください'
contexts = [
    {
        'text': '織田信長（1534-1582）は戦国時代の武将で、天下統一を目指した。',
        'article_title': '織田信長',
        'chunk_id': 1
    }
]

answer = generator.generate_answer(question, contexts)
print('Generated answer:')
print(answer)
"
```

## メモリ最適化
```python
# モデル量子化（メモリ節約）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # FP16使用
    device_map="auto"           # 自動配置
)
```

## 次のステップ
Issue #09: 検索・生成統合スクリプト（rag_wiki.py）の実装