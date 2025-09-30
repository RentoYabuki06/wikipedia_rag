# Issue #10: Gradio UIの実装（app_wiki.py）

**予想時間**: 30分
**難易度**: 中級
**学習項目**: Gradio、Web UI設計、ユーザビリティ

## 概要
RAG機能をWebブラウザで利用できるGradio UIを実装し、ローカル環境とHugging Face Spacesでの動作を可能にする。

## タスク

### 1. Gradio アプリケーションの実装 (`src/app_wiki.py`)
- Gradioインターフェースの作成
- RAGパイプラインとの統合
- リアルタイム質問応答機能
- エラーハンドリングとユーザーフィードバック

### 2. UI コンポーネント設計
- **入力**: テキストボックス（質問入力）
- **出力**: マークダウン形式の回答表示
- **設定**: 詳細オプション（top-k, 再ランク有無など）
- **状態表示**: 処理状況のインジケーター

### 3. ユーザビリティ機能
- サンプル質問ボタン
- 処理時間の表示
- エラーメッセージの適切な表示
- 参照情報の見やすい整形

### 4. デプロイ対応
- 環境変数による設定変更
- Hugging Face Spaces向けの最適化
- リソース制限への対応

## 実装例のスケルトン
```python
import gradio as gr
import logging
import os
import time
from pathlib import Path
import traceback

from rag_wiki import WikiRAG

# グローバルRAGインスタンス
rag_system = None

def initialize_rag():
    """RAGシステムを初期化"""
    global rag_system
    try:
        artifacts_dir = os.getenv("ARTIFACTS_DIR", "artifacts")
        rag_system = WikiRAG(artifacts_dir)
        return "✅ システムが正常に初期化されました"
    except Exception as e:
        error_msg = f"❌ システム初期化エラー: {e}"
        logging.error(error_msg)
        return error_msg

def process_question(question: str, top_k: int = 16, top_n: int = 5, 
                    use_rerank: bool = True, show_sources: bool = True) -> tuple:
    """質問を処理して回答を生成"""
    if not question.strip():
        return "質問を入力してください。", "", 0.0
    
    if rag_system is None:
        return "❌ システムが初期化されていません。", "", 0.0
    
    try:
        start_time = time.time()
        
        # RAG処理実行
        result = rag_system.search_and_generate(
            question=question,
            top_k=top_k,
            top_n=top_n,
            use_rerank=use_rerank
        )
        
        processing_time = time.time() - start_time
        
        # 回答の整形
        answer = result['answer']
        
        # ソース情報の整形
        sources_info = ""
        if show_sources and result['contexts']:
            sources_info = "\\n\\n### 📚 参照情報\\n"
            for i, ctx in enumerate(result['contexts']):
                title = ctx.get('article_title', 'Unknown')
                chunk_id = ctx.get('chunk_id', 0)
                score = ctx.get('vector_score', 0)
                sources_info += f"- **[{i}]** {title} (chunk {chunk_id}, スコア: {score:.3f})\\n"
        
        # 統計情報
        stats = result.get('search_stats', {})
        stats_info = f"""
### 📊 処理統計
- 処理時間: {processing_time:.2f}秒
- 検索候補数: {stats.get('total_candidates', 0)}
- 最終候補数: {stats.get('final_candidates', 0)}
- 再ランク使用: {'はい' if stats.get('rerank_used', False) else 'いいえ'}
"""
        
        full_response = answer + sources_info + stats_info
        
        return full_response, "", processing_time
        
    except Exception as e:
        error_msg = f"❌ 処理エラー: {e}"
        logging.error(f"Question processing error: {e}")
        logging.error(traceback.format_exc())
        return error_msg, "", 0.0

def create_sample_questions():
    """サンプル質問のリスト"""
    return [
        "大政奉還について教えてください",
        "織田信長の主な業績は何ですか？",
        "江戸幕府はいつ成立しましたか？",
        "明治維新の主な出来事を説明してください",
        "戦国時代の特徴について教えてください"
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
        gr.Markdown("""
        # 🔍 Wikipedia RAG システム
        
        日本語Wikipediaを使った質問応答システムです。質問を入力すると、関連する情報を検索して回答します。
        
        ⚠️ **注意**: 初回起動時はモデルの読み込みに時間がかかる場合があります。
        """)
        
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
                    elem_classes=["question-box"]
                )
                
                # サンプル質問ボタン
                with gr.Row():
                    sample_buttons = []
                    for i, sample_q in enumerate(create_sample_questions()[:3]):
                        btn = gr.Button(f"例{i+1}", size="sm")
                        sample_buttons.append((btn, sample_q))
                
                # 詳細設定
                with gr.Accordion("詳細設定", open=False):
                    top_k = gr.Slider(
                        minimum=5, maximum=50, value=16, step=1,
                        label="検索候補数 (top-k)"
                    )
                    top_n = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1,
                        label="最終候補数 (top-n)"
                    )
                    use_rerank = gr.Checkbox(
                        value=True, label="再ランキングを使用"
                    )
                    show_sources = gr.Checkbox(
                        value=True, label="参照情報を表示"
                    )
                
                # 実行ボタン
                submit_btn = gr.Button("🔍 質問する", variant="primary", size="lg")
            
            with gr.Column(scale=3):
                # 回答表示
                answer_output = gr.Markdown(
                    label="回答",
                    value="質問を入力して「🔍 質問する」ボタンを押してください。"
                )
        
        # フッター
        gr.Markdown("""
        ---
        ### 📖 使い方
        1. 左側の入力欄に質問を入力
        2. 「🔍 質問する」ボタンをクリック
        3. 右側に回答と参照情報が表示されます
        
        ### 💡 ヒント
        - 具体的な人名や事件名を含む質問が効果的です
        - 「〜について教えて」「〜とは何ですか」等の形式がおすすめです
        """)
        
        # イベントハンドラー
        def set_sample_question(sample_text):
            return sample_text
        
        # サンプル質問ボタンのイベント
        for btn, sample_q in sample_buttons:
            btn.click(
                fn=lambda sq=sample_q: sq,
                outputs=question_input
            )
        
        # 質問処理のイベント
        submit_btn.click(
            fn=process_question,
            inputs=[question_input, top_k, top_n, use_rerank, show_sources],
            outputs=[answer_output, gr.State(), gr.State()]
        )
        
        # Enter キーでの送信
        question_input.submit(
            fn=process_question,
            inputs=[question_input, top_k, top_n, use_rerank, show_sources],
            outputs=[answer_output, gr.State(), gr.State()]
        )
    
    return demo

def main():
    """メイン関数"""
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
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
        server_name=host,
        server_port=port,
        share=False,  # Hugging Face Spacesでは自動的にshareされる
        show_error=True
    )

if __name__ == "__main__":
    main()
```

## UI設計のポイント

### レスポンシブデザイン
- 異なる画面サイズに対応
- モバイルフレンドリーなレイアウト

### ユーザビリティ
- 明確な操作手順の提示
- サンプル質問による使い方の例示
- エラー時の分かりやすいメッセージ

### パフォーマンス
- 初期化状態の明示
- 処理時間の表示
- 長時間処理時のインジケーター

## 受け入れ基準
- [ ] Gradioアプリが正常に起動する
- [ ] 質問入力と回答表示が機能する
- [ ] サンプル質問ボタンが動作する
- [ ] 詳細設定が適切に反映される
- [ ] エラーハンドリングが適切に動作する
- [ ] 参照情報が見やすく表示される

## 学習ポイント
- Gradioによる機械学習アプリ開発
- ユーザーインターフェース設計
- 非同期処理とユーザビリティ
- エラーハンドリングの重要性
- デプロイメント対応の考慮事項

## テスト方法
```bash
# ローカル起動
python src/app_wiki.py

# カスタム設定での起動
HOST=127.0.0.1 PORT=8080 python src/app_wiki.py
```

## Hugging Face Spaces対応
`app.py` をルートに配置：
```python
# app.py (Spaces用エントリーポイント)
import sys
from pathlib import Path

# srcディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent / "src"))

from app_wiki import main

if __name__ == "__main__":
    main()
```

## 次のステップ
Issue #11: 評価データセットの作成とRecall@K評価機能の実装