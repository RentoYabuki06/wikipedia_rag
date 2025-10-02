import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List, Dict, Any, Optional
import logging
from config import DEFAULT_GENERATOR_MODEL, GENERATION_MAX_NEW_TOKENS, MAX_CONTEXTS_FOR_PROMPT, MAX_CONTEXT_LENGTH


class QwenGenerator:
    def __init__(self, model_name: str = DEFAULT_GENERATOR_MODEL, use_gpu: bool = True) -> None:
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

        # pad_tokenが設定されていない場合はeos_tokenを使用
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.info("Set pad_token to eos_token")

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _setup_generation_config(self):
        """生成設定を初期化"""
        self.generation_config = GenerationConfig(
            max_new_tokens=64,  # 短く設定して高速化
            do_sample=False,  # グリーディ生成（高速・決定的）
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def generate_answer(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        """質問とコンテキストから回答を生成"""
        try:
            # プロンプト構築
            logging.info("Building prompt...")
            prompt = self._build_prompt(question, contexts)
            logging.info(f"Prompt length: {len(prompt)} characters")

            # 生成実行
            logging.info("Starting text generation...")
            answer = self._generate_text(prompt)
            logging.info(f"Generated answer length: {len(answer)} characters")

            # 参照情報を追加
            logging.info("Adding source references...")
            answer_with_sources = self._add_source_references(answer, contexts)

            return answer_with_sources

        except Exception as e:
            logging.error(f"Answer generaton failed: {e}")
            return "申し訳ありませんが、回答の生成中にエラーが発生しました"

    def _build_prompt(self, question: str, contexts: List[Dict[str, Any]]) -> str:
        """プロンプトを構築"""
        # 設定ファイルから値を取得
        limited_contexts = contexts[:MAX_CONTEXTS_FOR_PROMPT]
        context_str = format_contexts(limited_contexts, max_length=MAX_CONTEXT_LENGTH)

        # シンプルなプロンプト
        prompt = f"参考情報:\n{context_str}\n\n" f"質問: {question}\n" f"回答: "
        return prompt

    def _generate_text(self, prompt: str) -> str:
        """テキスト生成を実行"""
        logging.info("Tokenizing prompt...")

        # シンプルなトークン化（paddingなし）
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(self.device)
        input_length = input_ids.shape[1]
        logging.info(f"Input token count: {input_length}")

        # 入力が長すぎる場合は警告
        if input_length >= 512:
            logging.warning(f"Input length: {input_length}, truncated to 512")

        logging.info("Running model.generate()...")
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=GENERATION_MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            logging.info("✓ Generation completed")
        except Exception as e:
            logging.error(f"Generation failed: {e}")
            raise

        output_length = output_ids.shape[1]
        logging.info(f"Output token count: {output_length} (new: {output_length - input_length})")

        logging.info("Decoding output...")
        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        logging.info(f"Raw generated text length: {len(generated)}")

        # プロンプト部分を除去して回答のみ返す
        # "回答: " の後の部分だけを抽出
        if "回答: " in generated:
            answer = generated.split("回答: ", 1)[-1].strip()
        elif generated.startswith(prompt):
            answer = generated[len(prompt) :].strip()
        else:
            answer = generated.strip()

        logging.info(f"Final answer preview: {answer[:50]}...")
        return answer

    def _add_source_references(self, answer: str, contexts: List[Dict[str, Any]]) -> str:
        """回答に参照情報を追加"""
        if not contexts:
            return answer
        sources = [extract_source_info(ctx) for ctx in contexts]
        sources_str = "\n\n【参照元】\n" + "\n".join(sources)
        return answer + sources_str


def format_contexts(contexts: List[Dict[str, Any]], max_length: int = 200) -> str:
    """コンテキストを表示的に整形"""
    formatted = []
    for i, ctx in enumerate(contexts):
        text = ctx["text"]
        # 各コンテキストを制限長に切り詰め
        if len(text) > max_length:
            text = text[:max_length] + "..."
        formatted.append(f"[{i}] {text}")
    return "\n\n".join(formatted)


def extract_source_info(context: Dict[str, Any]) -> str:
    """ソース情報を抽出"""
    return f"jawiki:{context.get('article_title', 'Unknown')}#chunk={context.get('chunk_id', 0)}"
