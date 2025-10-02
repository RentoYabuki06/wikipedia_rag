import gradio as gr
import logging
import os
import time
from pathlib import Path
import traceback

from rag_wiki import WikiRAG
from config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_TOP_K,
    DEFAULT_TOP_N,
    USE_RERANKER,
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_EMBEDDER_MODEL,
    DEFAULT_RERANKER_MODEL,
)

# グローバルRAGインスタンス
rag_system = None


def initialize_rag():
    """RAGシステムを初期化"""
    global rag_system
    try:
        artifacts_dir = os.getenv("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR)
        logging.info(f"Initializing RAG system with artifacts_dir: {artifacts_dir}")
        rag_system = WikiRAG(artifacts_dir)
        return "✅ システムが正常に初期化されました"
    except Exception as e:
        error_msg = f"❌ システム初期化エラー: {e}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        return error_msg


def process_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    top_n: int = DEFAULT_TOP_N,
    use_rerank: bool = USE_RERANKER,
    show_sources: bool = True,
) -> str:
    """質問を処理して回答を生成"""
    if not question.strip():
        return "質問を入力してください。"

    if rag_system is None:
        return "❌ システムが初期化されていません。"

    try:
        start_time = time.time()

        # RAG処理実行
        result = rag_system.search_and_generate(question=question, top_k=top_k, top_n=top_n, use_rerank=use_rerank)

        processing_time = time.time() - start_time

        # 回答の整形
        answer = result["answer"]

        # ソース情報の整形
        sources_info = ""
        if show_sources and result["contexts"]:
            sources_info = "\n\n### 📚 参照情報\n"
            for i, ctx in enumerate(result["contexts"]):
                title = ctx.get("article_title", "Unknown")
                chunk_id = ctx.get("chunk_id", 0)
                score = ctx.get("vector_score", 0)
                sources_info += f"- **[{i}]** {title} (chunk {chunk_id}, スコア: {score:.3f})\n"

        # 統計情報
        stats = result.get("search_stats", {})
        stats_info = f"""
### 📊 処理統計
- 処理時間: {processing_time:.2f}秒
- 検索候補数: {stats.get('total_candidates', 0)}
- 最終候補数: {stats.get('final_candidates', 0)}
- 再ランク使用: {'はい' if stats.get('rerank_used', False) else 'いいえ'}
"""

        # モデル情報
        try:
            generator_model = rag_system.generator.model_name
            device = str(rag_system.generator.device)
            reranker_status = "有効" if use_rerank and rag_system.reranker.is_available else "無効"
        except:
            generator_model = DEFAULT_GENERATOR_MODEL
            device = "CPU"
            reranker_status = "無効" if not use_rerank else "不明"

        model_info = f"""
### 🤖 使用モデル
- **生成モデル**: `{generator_model}`
- **埋め込みモデル**: `{DEFAULT_EMBEDDER_MODEL}`
- **再ランクモデル**: `{DEFAULT_RERANKER_MODEL}` ({reranker_status})
- **デバイス**: {device}
"""

        full_response = answer + sources_info + stats_info + model_info

        return full_response

    except Exception as e:
        error_msg = f"❌ 処理エラー: {e}"
        logging.error(f"Question processing error: {e}")
        logging.error(traceback.format_exc())
        return error_msg


def create_sample_questions():
    """サンプル質問のリスト"""
    return [
        "アンパサンドとは何ですか？",
        "言語の定義について教えてください",
        "ソクラテスの思想について説明してください",
    ]


def create_interface():
    """Gradioインターフェースを作成"""

    # カスタムCSS
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .question-box textarea {
        font-size: 16px !important;
    }
    """

    with gr.Blocks(css=css, title="Wikipedia RAG システム", theme=gr.themes.Soft()) as demo:

        # ヘッダー
        gr.Markdown(
            """
        # 🔍 Wikipedia RAG システム
        
        日本語Wikipediaを使った質問応答システムです。質問を入力すると、関連する情報を検索して回答します。
        
        ⚠️ **注意**: 初回起動時はモデルの読み込みに時間がかかる場合があります。
        """
        )

        # システム状態表示
        with gr.Row():
            status_display = gr.Markdown("🔄 システムを初期化しています...")

        # メインインターフェース
        with gr.Row():
            with gr.Column(scale=2):
                # 質問入力
                question_input = gr.Textbox(
                    label="質問を入力してください",
                    placeholder="例: 織田信長について教えてください",
                    lines=3,
                    elem_classes=["question-box"],
                )

                # サンプル質問ボタン
                with gr.Row():
                    sample_buttons = []
                    for i, sample_q in enumerate(create_sample_questions()[:3]):
                        btn = gr.Button(f"例{i+1}", size="sm")
                        sample_buttons.append((btn, sample_q))

                # 詳細設定
                with gr.Accordion("詳細設定", open=False):
                    top_k = gr.Slider(minimum=5, maximum=50, value=DEFAULT_TOP_K, step=1, label="検索候補数 (top-k)")
                    top_n = gr.Slider(minimum=1, maximum=10, value=DEFAULT_TOP_N, step=1, label="最終候補数 (top-n)")
                    use_rerank = gr.Checkbox(value=USE_RERANKER, label="再ランキングを使用")
                    show_sources = gr.Checkbox(value=True, label="参照情報を表示")

                # 実行ボタン
                submit_btn = gr.Button("🔍 質問する", variant="primary", size="lg")

            with gr.Column(scale=3):
                # 回答表示
                answer_output = gr.Markdown(label="回答", value="質問を入力して「🔍 質問する」ボタンを押してください。")

        # フッター
        gr.Markdown(
            """
        ---
        ### 📖 使い方
        1. 左側の入力欄に質問を入力
        2. 「🔍 質問する」ボタンをクリック
        3. 右側に回答と参照情報が表示されます
        
        ### 💡 ヒント
        - インデックスに含まれる記事に関する質問が効果的です
        - サンプル質問のボタンで試してみてください
        - 「〜について教えて」「〜とは何ですか」等の形式がおすすめです
        """
        )

        # イベントハンドラー
        def set_sample_question(sample_text):
            return sample_text

        # サンプル質問ボタンのイベント
        for btn, sample_q in sample_buttons:
            btn.click(fn=lambda sq=sample_q: sq, outputs=question_input)

        # 質問処理のイベント
        submit_btn.click(
            fn=process_question, inputs=[question_input, top_k, top_n, use_rerank, show_sources], outputs=answer_output
        )

        # Enter キーでの送信
        question_input.submit(
            fn=process_question, inputs=[question_input, top_k, top_n, use_rerank, show_sources], outputs=answer_output
        )

    return demo


def main():
    """メイン関数"""
    # ログ設定
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 環境設定
    port = int(os.getenv("PORT", 7860))
    host = os.getenv("HOST", "0.0.0.0")

    # RAGシステム初期化
    init_result = initialize_rag()
    logging.info(init_result)

    # Gradioアプリ作成・起動
    demo = create_interface()

    logging.info(f"Starting Gradio app on {host}:{port}")

    demo.launch(
        server_name=host, server_port=port, share=False, show_error=True  # Hugging Face Spacesでは自動的にshareされる
    )


if __name__ == "__main__":
    main()
