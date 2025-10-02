# Wikipedia RAG システム

日本語Wikipediaを使った検索拡張生成（RAG）システムです。質問を入力すると、関連するWikipedia記事を検索し、その情報に基づいてAIが回答を生成します。

## ✨ できること

- 💬 **自然な日本語で質問**: 「アンパサンドって何？」「ソクラテスについて教えて」など
- � **正確な情報検索**: Wikipedia記事から関連情報を高速検索
- 📝 **出典付き回答**: すべての回答に参照元の記事情報を表示
- 🖥️ **使いやすいWeb UI**: ブラウザから簡単にアクセス可能

## 🚀 5分で始める

### 1. インストール

```bash
# リポジトリをクローン
git clone https://github.com/RentoYabuki06/wikipedia_rag.git
cd wikipedia_rag

# 必要なパッケージをインストール
pip install -r requirements.txt
```

### 2. Wikipediaインデックスを構築

```bash
# 3000記事でインデックスを作成（約5-10分）
python src/build_wiki_index.py --max_articles 3000
```

> 💡 **ヒント**: 記事数を増やすと検索範囲が広がりますが、処理時間も増加します
> - 小規模テスト: `--max_articles 1000`
> - 標準: `--max_articles 3000`（推奨）
> - 大規模: `--max_articles 10000`（30分以上）

### 3. Web UIを起動

```bash
# Gradio Web UIを起動
python src/app_wiki.py
```

ブラウザで `http://localhost:7860` を開くと、すぐに使えます！

### 4. コマンドラインから使う（オプション）

```bash
# 直接質問して回答を取得
python src/rag_wiki.py -q "アンパサンドについて教えて"
```

## 💡 使い方の例

### Web UI での質問

1. `python src/app_wiki.py` でUIを起動
2. ブラウザで質問を入力（例: 「アンパサンドについて教えて」）
3. 約10秒で回答が表示されます

**サンプル質問:**
- 「言語とは何ですか？」
- 「ソクラテスについて教えて」
- 「アンパサンドの意味は？」

### コマンドラインでの質問

```bash
$ python src/rag_wiki.py -q "言語について教えて"

==================================================
質問: 言語について教えて
==================================================
回答: 言語とは、人間が意思や感情を伝達するために用いる体系的な記号システムです。
音声言語と文字言語があり、各地域や文化によって異なる特徴を持ちます...

📚 参照元:
[0] jawiki:言語#chunk=0 (類似度: 0.892)
[1] jawiki:日本語#chunk=1 (類似度: 0.854)

⏱️ 処理時間: 10.2秒
==================================================
```

### 検索精度の調整

```bash
# より多くの記事を検索（デフォルト: 5）
python src/rag_wiki.py -q "質問" --top_k 10

# 再ランキングを有効化（精度向上、処理時間増加）
python src/rag_wiki.py -q "質問" --use_rerank

# 使用する文脈数を変更（デフォルト: 3）
python src/rag_wiki.py -q "質問" --top_n 5
```

## 🎯 技術仕様

### 使用モデル

| 用途 | モデル | 説明 |
|------|--------|------|
| **埋め込み** | `intfloat/multilingual-e5-base` | 多言語対応の意味ベクトル化モデル |
| **生成** | `Qwen/Qwen2.5-0.5B-Instruct` | 超高速軽量な対話型言語モデル |
| **再ランク** | `BAAI/bge-reranker-v2-m3` | 高精度な文書ランキングモデル（オプション） |

### パフォーマンス

- **検索速度**: 約1秒（FAISS使用）
- **生成速度**: 約3-6秒（0.5Bモデル）
- **合計応答時間**: 約10-15秒（再ランク無効時）
- **メモリ使用量**: 約4-6GB（CPU使用時）

### システム構成

```
質問入力
  ↓
埋め込み化（E5モデル）
  ↓
ベクトル検索（FAISS）
  ↓
再ランキング（BGE）※オプション
  ↓
文脈生成（上位記事を結合）
  ↓
回答生成（Qwen2.5モデル）
  ↓
回答＋出典表示
```

## 📋 システム要件

- **Python**: 3.10以上
- **RAM**: 8GB以上推奨（最低4GB）
- **ストレージ**: 10GB以上の空き容量
- **GPU**: 不要（CPUで動作、GPUがあれば高速化）

## ⚙️ カスタマイズ

すべての設定は `src/config.py` で一括管理されています：

```python
# モデルの変更
DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # より大きなモデルに変更可能

# 検索パラメータ
DEFAULT_TOP_K = 5        # 検索する記事数
DEFAULT_TOP_N = 3        # 回答に使用する記事数

# 生成パラメータ  
GENERATION_MAX_NEW_TOKENS = 200  # 回答の最大トークン数

# 再ランカー
USE_RERANKER = False     # 高精度化したい場合はTrue（処理時間増加）
```

設定変更後は再起動なしで反映されます。

## 🔧 トラブルシューティング

### インデックスが見つからない

```bash
# artifactsディレクトリを確認
ls artifacts/

# なければインデックスを再構築
python src/build_wiki_index.py --max_articles 3000
```

### メモリエラーが出る

```python
# src/config.py でモデルサイズを小さく
DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # より小さいモデル

# または検索数を減らす
DEFAULT_TOP_K = 3
DEFAULT_TOP_N = 2
```

### 応答が遅い

```python
# src/config.py で再ランカーを無効化
USE_RERANKER = False  # これで約100秒短縮

# 生成トークン数を減らす
GENERATION_MAX_NEW_TOKENS = 100
```

### 特定の記事が見つからない

```bash
# より多くの記事でインデックスを再構築
python src/build_wiki_index.py --max_articles 10000
```

## � 開発者向け情報

### プロジェクト構成

```
wikipedia_rag/
├── src/
│   ├── config.py           # 設定ファイル（モデル、パラメータ）
│   ├── data_loader.py      # Wikipedia データ取得
│   ├── chunker.py          # テキスト分割処理
│   ├── embedder.py         # 埋め込みベクトル生成
│   ├── vector_store.py     # FAISSインデックス管理
│   ├── reranker.py         # 再ランキング処理
│   ├── generator.py        # LLM回答生成
│   ├── rag_wiki.py         # RAGパイプライン統合
│   ├── build_wiki_index.py # インデックス構築スクリプト
│   └── app_wiki.py         # Gradio Web UI
├── artifacts/              # 生成データ（インデックス等）
├── docs/                   # ドキュメント
│   └── plan.md            # 開発計画
└── requirements.txt        # 依存パッケージ
```

### 学習用リソース

このプロジェクトは学習用に段階的実装ができるよう設計されています。

**12ステップの学習パス:**
1. プロジェクトセットアップ
2. Wikipediaデータ取得
3. テキスト分割（chunking）
4. 埋め込みモデル実装
5. FAISSインデックス構築
6. 検索パイプライン統合
7. 再ランキング実装
8. LLM生成実装
9. RAG統合
10. Gradio UI実装
11. 評価機能実装
12. 統合テスト・ドキュメント

詳細は [docs/github-workflow-guide.md](docs/github-workflow-guide.md) を参照してください。

### テスト

```bash
# ジェネレータ単体テスト
python test_generator_only.py

# RAG統合テスト
python src/rag_wiki.py -q "テスト質問"

# Web UIテスト
python src/app_wiki.py
```

## 🤝 コントリビューション

プルリクエストを歓迎します！以下の手順でコントリビュートできます：

1. このリポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## 📝 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🙏 謝辞

このプロジェクトは以下のオープンソースプロジェクトを使用しています：

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FAISS](https://github.com/facebookresearch/faiss) by Meta Research
- [Gradio](https://gradio.app/)
- [Wikipedia](https://www.wikipedia.org/) contributors

## � 問い合わせ

質問や提案がある場合：

- 💭 [Discussions](../../discussions) で相談
- 🐛 [Issues](../../issues) でバグ報告
- 📧 [@RentoYabuki06](https://github.com/RentoYabuki06) にメンション

---

**今すぐ使ってみる:**

```bash
git clone https://github.com/RentoYabuki06/wikipedia_rag.git
cd wikipedia_rag
pip install -r requirements.txt
python src/build_wiki_index.py --max_articles 3000
python src/app_wiki.py
```

楽しいWikipedia探索を！ 🎉📚