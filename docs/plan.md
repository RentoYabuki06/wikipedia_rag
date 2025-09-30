# 日本語WikipediaミニRAG 要件定義書

最終更新: 2025-09-30 (JST)
作成者: あなた（開発者）
対象: LLM/エンジニア/将来のメンテナ

---

## 1. 目的・ゴール

* **目的**: コーパス準備なしで、日本語Wikipediaのサブセットを用いた最小構成のRAG（Retrieval Augmented Generation）を構築し、質問応答を日本語で返す。
* **ゴール**:

  * Wikipedia(ja)データセットの自動取得→分割→埋め込み→FAISS構築→検索→（任意再ランキング）→LLM生成までを**ワンコマンド**で再現可能。
  * 回答末尾に **出典（タイトル/チャンクID）** を列挙する。
  * Gradio UIでローカル動作 & Hugging Face Spacesにそのままデプロイ可能。

## 2. スコープ

* **In**: 日本語WikipediaベースのRAGパイプライン、CLI、最小評価（Recall@K）、Gradio UI。
* **Out**: 外部ベクタDB（Pinecone等）、高度なガバナンス、監視基盤、PDF/Webクロール。

## 3. 前提・依存

* Python 3.10+、pip。
* 主要ライブラリ: `transformers`, `datasets`, `sentence-transformers`, `faiss-cpu`, `accelerate`, `gradio`。
* モデル（デフォルト）:

  * **埋め込み**: `intfloat/multilingual-e5-base`
  * **再ランキング(任意)**: `BAAI/bge-reranker-v2-m3`
  * **LLM**: `Qwen/Qwen2.5-3B-Instruct`（ローカル/GPU前提。重い場合は別モデルへ変更可）

## 4. 入出力仕様

### 入力

* ユーザ質問（日本語想定／英語も可）
* 内部入力: Wikipedia(ja)データセット（例: `wikipedia`, config=`20231101.ja`）

### 出力

* 日本語の回答テキスト。
* 末尾に **参照一覧**（`source=jawiki:<title>#chunk=<n>` を `[0],[1],...` で対応付け）。
* 参照ゼロ時は「該当コンテキストが見つかりませんでした。」等の明示的な不確実性表明。

## 5. 機能要件

1. **コーパス取得**: `datasets.load_dataset("wikipedia", "20231101.ja", split="train")` を使い、最大 `MAX_ARTICLES` 件を取り込み。
2. **正規化/分割**:

   * 正規化: 連続空白を単一化、前後空白除去。
   * チャンク: 文字ベース `size=450`, `overlap=60`（日本語向け）。
   * メタ: `id`, `source=jawiki:<title>`, `chunk_id`, `text` を `*.jsonl` に保存。
3. **埋め込み/索引化**:

   * E5 前置詞: passage側に `"passage: "`、query側に `"query: "` を**必ず**付与。
   * ベクトルはL2正規化し、FAISS `IndexFlatIP` に登録。
4. **検索**:

   * まず `k=16` 取得、スコア降順で `n=5` を最終候補に。
   * （任意）クロスエンコーダ再ランキング（`BAAI/bge-reranker-v2-m3`）でTop-Nを並べ替え。
5. **プロンプト生成/回答**:

   * **System**: 事実重視・推測禁止・根拠引用を指示。
   * **Context**: 取得チャンクを `[i]` で連番表示。
   * **Instruction**: 「コンテキストの範囲内のみで回答」「不明は不明と答える」「最後に参照IDを列挙」。
   * **出力**: 本文 + 空行 + `参照: [0] jawiki:...#chunk=..., [1] ...`
6. **UI**:

   * Gradio Blocks: テキスト入力→回答テキスト出力。タイトル・注意書きの表示。
7. **CLI**:

   * `python build_wiki_index.py`（コーパス→索引）
   * `python rag_wiki.py --q "質問"`（1ショット応答）
   * `python app_wiki.py`（UI起動）
8. **評価（最小）**:

   * `data/dev.jsonl` に質問と `gold_sources`（`source#chunk` の配列）を持つ開発セットを用意。
   * `Recall@K`（K=5）を算出するスクリプトを提供。

## 6. 非機能要件

* **再現性**: 乱数固定、バージョンを `requirements.txt` にピン留め（可能な範囲）。
* **性能**: 1,000〜3,000記事（数万チャンク）で数秒以内の検索応答（CPU/メモリ依存）。
* **可搬性**: ローカル/Spacesの双方で動作。
* **可観測性**: ログ出力（読み込み件数、索引サイズ、検索ヒット件数、再ランク有無）。

## 7. データ構造

* `artifacts/wiki_metas.jsonl`: 1行1レコード `{id, source, chunk_id, text}`
* `artifacts/wiki.index`: FAISS IndexFlatIP
* `data/dev.jsonl`: `{question: str, gold_sources: ["jawiki:<title>#<cid>", ...]}`

## 8. プロンプト設計（雛形）

* **system**: 「あなたは事実に忠実なアシスタントです。与えられたコンテキスト以外は推測せず、根拠を引用して回答してください。」
* **user**:

  * 質問（プレーン）
  * 参考コンテキスト（`[0] ...` の形で複数）
  * 指示: 「コンテキストのみで回答」「不明時は不明」「最後に参照ID列挙」

> 注意: LLMが参照IDを勝手改変しないよう、文中参照は `[0],[1]...` のみを許可。末尾に人間可読の `source#chunk` リストを付与。

## 9. 失敗モードとハンドリング

* 参照ゼロ（検索未ヒット）: 「該当コンテキストが見つかりませんでした。」
* LLM hallucination疑い: 出典ゼロ回答は禁止。テンプレで参照付与を強制。
* 再ランカーモデル未取得/エラー: フォールバックでベクトル類似度順を使用。

## 10. セキュリティ/ライセンス

* WikipediaデータはCreative Commons Attribution-ShareAlike 3.0。出力で出典を示し、再配布ポリシーに留意。
* モデルのライセンス（Qwen等）は各レポを確認。商用時は差し替え可能に設計。

## 11. テスト計画

* 単体: 正規化・分割境界（オーバーラップ）、E5前置詞の有無での検索差。
* 結合: `build→search→answer` ひと通りが成功すること。
* 評価: devセットで `Recall@5 ≥ 0.6`（小規模条件の暫定基準、調整可）。

## 12. ディレクトリ/ファイル構成

```
project/
  data/
    dev.jsonl                      # 開発用評価データ（任意）
  artifacts/
    wiki_metas.jsonl
    wiki.index
  src/
    build_wiki_index.py            # 取得→分割→埋め込み→FAISS
    rag_wiki.py                    # 検索+生成（CLI callable）
    app_wiki.py                    # Gradio UI
    eval_retrieval.py              # Recall@K
  requirements.txt
  README.md
```

## 13. 設定パラメータ（既定値）

* `WIKI_CONFIG = "20231101.ja"`（fallback: `20220301.ja`）
* `MAX_ARTICLES = 30000`
* Chunking: `size=450`, `overlap=60`
* Retrieval: `k=16`, Final: `n=5`
* Models:

  * Embed: `intfloat/multilingual-e5-base`
  * Reranker (optional): `BAAI/bge-reranker-v2-m3`
  * LLM: `Qwen/Qwen2.5-3B-Instruct`

## 14. CLI I/F 仕様

* 索引作成: `python src/build_wiki_index.py --max 30000 --config 20231101.ja`

  * オプション: `--chunk_size`, `--overlap`, `--out_dir`
* QA一発: `python src/rag_wiki.py --q "大政奉還の概要を一言で。" --topk 16 --topn 5 --no-rerank`
* UI: `python src/app_wiki.py --host 0.0.0.0 --port 7860`

## 15. 受け入れ基準（Acceptance Criteria）

1. クリーン環境で `pip install -r requirements.txt` 後、3コマンドで**索引→QA→UI**が通る。
2. LLM出力の末尾に `[0]...[n]` 参照と `source#chunk` が表示される。
3. `data/dev.jsonl` を用いた `Recall@5` が実行可能で、閾値（例: ≥0.6）を満たすか、原因と改善案がログに残る。
4. 環境変数/CLIでモデル差し替えが可能（少なくとも Embed/LLM）。

## 16. 拡張案（ロードマップ）

* **HyDE**: 仮想回答生成→それをクエリ埋め込み。
* **bge-m3**: BM25 + Dense + Multi-Vector のハイブリッド化。
* **親子チャンク**: セクション単位の親→文単位の子で再スコア。
* **応答テンプレ**: 箇条書き/要約/定義/年表モードなどスタイル切替。
* **キャッシュ**: `q→doc_ids` と `q,doc_ids→answer` の二段キャッシュ。
* **Spaces公開**: `app_wiki.py` をそのままデプロイ、`README.md` に使い方記載。

## 17. リスク・制約

* LLMサイズによりレイテンシ/VRAMが増大。軽量モデルやHF Inference API利用を検討。
* Wikipediaの最新性はダンプ日付に依存。最新版でない情報の可能性に注意。
* 再ランカーモデルはGPUが無い環境で遅い/使えない場合あり。

## 18. サンプル（I/O）

* **Input**: 「薩長同盟の成立年を教えて」
* **検索ヒット例**: `[0] ...1866年（慶応2年）...`
* **Output（要旨）**:

  * 本文: 「1866年（慶応2年）です。...」
  * 参照: `[0] jawiki:薩長同盟#chunk=3, [1] jawiki:坂本龍馬#chunk=5`

---

### 付録A: 例外メッセージ

* 「該当コンテキストが見つかりませんでした。質問を言い換えるか、より一般的な表現を試してください。」

### 付録B: ログ例

* `Loaded 28,500 articles → 92,300 chunks`
* `FAISS ntotal=92300, dim=768`
* `Search k=16 -> rerank=True -> topn=5`
