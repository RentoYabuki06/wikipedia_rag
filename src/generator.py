import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List, Dict, Any, Optional
import logging


class QwenGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", use_gpu: bool = True) -> None:
        """Qwen生成モデルを初期化"""
        self.model_name = model_name
        self.device = self._setup_device(use_gpu)
        self.tokenizer = None
        self.model = None
        self.generation_config = None

        try:
            self._load_model()
            self._setup_generation_config()  # ここで生成設定を初期化
            logging.info(f"Generator loaded successfully: {model_name}")
        except Exception as e:
            logging.error(f"Failed to load generator: {e}")
            raise

    def _setup_device(self, use_gpu: bool) -> torch.device:
        """デバイスを設定"""
        if use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _load_model(self):
        """モデルとトークナイザーを読み込み"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _setup_generation_config(self):
        """生成設定を初期化"""
        self.generation_config = GenerationConfig(
            max_new_token=512,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
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
            logging.error(f"Answer generaton failed: {e}")
            return "申し訳ありませんが、回答の生成中にエラーが発生しました"

    def _build_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        """プロンプトを構築"""
        context_str = format_contexts(contexts)
        prompt = (
            "以下は質問応答タスクです\n"
            "与えられた文脈を参考にして、質問日本語で回答してください\n"
            f"【文脈】\n{context_str}\n\n"
            f"【質問】\n{question}\n\n"
            "【回答】"
        )
        return prompt

    def _generate_text(self, prompt: str) -> str:
        """テキスト生成を実行"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, generation_config=self.generation_config)
        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # プロンプト部分を除去して回答のみ返す
        if generated.startswith(prompt):
            return generated[len(prompt) :].strip()
        return generated.strip()

    def _add_source_references(self, answer: str, contexts: List[Dict[str, Any]]) -> str:
        """回答に参照情報を追加"""
        if not contexts:
            return answer
        sources = [extract_source_info(ctx) for ctx in contexts]
        sources_str = "\n\n【参照元】\n" + "\n".join(sources)
        return answer + sources_str


def format_contexts(contexts: List[Dict[str, Any]]) -> str:
    """コンテキストを表示的に整形"""
    formatted = []
    for i, ctx in enumerate(contexts):
        formatted.append(f"[{i}] {ctx['text']}")
    return "\n\n".join(formatted)


def extract_source_info(context: Dict[str, Any]) -> str:
    """ソース情報を抽出"""
    return f"jawiki:{context.get('article_title', 'Unknown')}#chunk={context.get('chunk_id', 0)}"
