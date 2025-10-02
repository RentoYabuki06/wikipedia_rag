"""
言語モデル単体のテスト
RAG、ベクトル検索、rerankをスキップして、生成モデルだけを確認
"""

import sys
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# src/ をパスに追加
sys.path.insert(0, str(Path(__file__).parent / "src"))


def setup_logging():
    """ログ設定"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def test_raw_model():
    """transformersライブラリを直接使用した最小限のテスト"""
    print("=" * 60)
    print("RAW モデルテスト (transformers直接使用)")
    print("=" * 60)

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # 最軽量版でテスト

    print(f"\n[1] モデル読み込み中: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # pad_token設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  - pad_token を eos_token に設定")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    print("✓ モデル読み込み完了\n")

    # 超シンプルなプロンプト
    prompt = "東京は"
    print(f"[2] プロンプト: '{prompt}'")

    # トークン化
    print("[3] トークン化中...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    print(f"  - 入力トークン数: {input_length}")

    # 生成（最小限のパラメータ）
    print("[4] 生成中 (max_new_tokens=10)...")
    try:
        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        print("✓ 生成完了")
    except Exception as e:
        print(f"✗ 生成失敗: {e}")
        return False

    # デコード
    print("[5] デコード中...")
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("結果")
    print("=" * 60)
    print(f"プロンプト: {prompt}")
    print(f"生成結果: {generated}")
    print("=" * 60)

    if len(generated) > len(prompt):
        print("\n✓ 成功: 新しいテキストが生成されました")
        return True
    else:
        print("\n✗ 失敗: 新しいテキストが生成されませんでした")
        return False


def test_generator_class():
    """QwenGeneratorクラスのテスト"""
    print("\n" + "=" * 60)
    print("[STEP 2] QwenGenerator クラステスト")
    print("=" * 60)

    from generator import QwenGenerator

    print("\n[1] QwenGenerator初期化中...")
    generator = QwenGenerator(model_name="Qwen/Qwen2.5-0.5B-Instruct")
    print("✓ 初期化完了")

    # ダミーコンテキスト
    contexts = [{"text": "織田信長は戦国時代の武将。", "article_title": "テスト", "chunk_id": 0}]

    question = "織田信長について教えて"

    print(f"\n[2] 質問: {question}")
    print("[3] 回答生成中...")

    try:
        answer = generator.generate_answer(question, contexts)

        print("\n" + "=" * 60)
        print("結果")
        print("=" * 60)
        print(f"質問: {question}")
        print(f"回答: {answer}")
        print("=" * 60)

        if answer and len(answer) > 0:
            print("\n✓ 成功: QwenGeneratorが正常に動作しました")
            return True
        else:
            print("\n✗ 失敗: 回答が空です")
            return False

    except Exception as e:
        print(f"\n✗ エラー: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    setup_logging()

    try:
        # STEP 1: RAWモデルテスト
        print("\n[STEP 1] transformers直接テスト")
        success_raw = test_raw_model()

        if not success_raw:
            print("\n❌ 基本的な生成が動作しません。モデルまたは環境に問題があります。")
            return 1

        print("\n✓ STEP 1 成功！")

        # STEP 2: QwenGeneratorクラステスト
        success_generator = test_generator_class()

        if not success_generator:
            print("\n❌ QwenGeneratorクラスに問題があります")
            return 1

        print("\n" + "=" * 60)
        print("🎉 全テスト成功！")
        print("=" * 60)
        print("✓ transformers直接テスト: PASS")
        print("✓ QwenGeneratorクラス: PASS")
        print("\nRAGパイプライン全体を試す準備ができました！")
        print("次のコマンドを実行してください:")
        print('  python src/rag_wiki.py -q "織田信長について教えて"')
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  ユーザーによって中断されました")
        return 130
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
