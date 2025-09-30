# GitHub Issues 自動作成ガイド

このリポジトリには、`issues/` フォルダ内のMarkdownファイルをGitHub Issuesに自動変換するワークフローが含まれています。

## 🚀 使用方法

### 1. 全てのIssueを一括作成

1. GitHubリポジトリページで **Actions** タブをクリック
2. **Convert Issues to GitHub Issues** ワークフローを選択
3. **Run workflow** ボタンをクリック
4. **Create all issues at once** にチェックを入れる
5. **Run workflow** を実行

### 2. 特定のIssueのみ作成

1. GitHubリポジトリページで **Actions** タブをクリック
2. **Convert Issues to GitHub Issues** ワークフローを選択  
3. **Run workflow** ボタンをクリック
4. **Specific issue numbers to create** に番号をカンマ区切りで入力（例: `01,02,03`）
5. **Run workflow** を実行

## 📋 作成されるIssue一覧

| Issue # | タイトル | 難易度 | 予想時間 | ラベル |
|---------|----------|--------|----------|---------|
| 01 | 🏗️ プロジェクトの基本セットアップ | 初級 | 15分 | `enhancement`, `good first issue`, `初級`, `setup` |
| 02 | 📥 Wikipediaデータ取得機能の実装 | 初級 | 20分 | `feature`, `初級`, `data-processing` |
| 03 | ✂️ テキスト分割（チャンキング）機能の実装 | 中級 | 25分 | `feature`, `中級`, `nlp`, `chunking` |
| 04 | 🔤 埋め込みモデル（E5）のセットアップと実装 | 中級 | 30分 | `feature`, `中級`, `embedding`, `e5` |
| 05 | 🔍 FAISSインデックスの構築と保存 | 中級 | 25分 | `feature`, `中級`, `vector-search`, `faiss` |
| 06 | ⚙️ インデックス構築用統合スクリプトの実装 | 中級 | 30分 | `feature`, `中級`, `pipeline`, `cli` |
| 07 | 🔄 再ランキング機能の実装（BGE Reranker） | 中級 | 25分 | `feature`, `中級`, `reranking`, `bge` |
| 08 | 🤖 LLM応答生成機能の実装（Qwen2.5） | 上級 | 30分 | `feature`, `上級`, `llm`, `generation`, `qwen` |
| 09 | 🔗 検索・生成統合スクリプトの実装 | 中級 | 25分 | `feature`, `中級`, `rag`, `integration` |
| 10 | 🖥️ Gradio UIの実装（app_wiki.py） | 中級 | 30分 | `feature`, `中級`, `ui`, `gradio`, `web` |
| 11 | 📊 評価データセットの作成とRecall@K評価機能 | 中級 | 25分 | `feature`, `中級`, `evaluation`, `testing` |
| 12 | 🧪 統合テスト・デバッグ・ドキュメント整備 | 中級 | 30分 | `testing`, `中級`, `documentation`, `integration` |

## 📊 プロジェクト管理

ワークフロー実行時に以下も自動作成されます：

### Project Board: "Wikipedia RAG Development"
- **ToDo**: 未着手のタスク
- **In Progress**: 作業中のタスク  
- **Review**: レビュー待ち
- **Done**: 完了したタスク

## 🏷️ ラベル説明

### 難易度ラベル
- `初級`: Python基本、環境構築レベル
- `中級`: ライブラリ統合、アルゴリズム実装レベル
- `上級`: システム設計、複雑な統合レベル

### 技術ラベル
- `setup`: 環境セットアップ関連
- `data-processing`: データ処理関連
- `nlp`: 自然言語処理関連
- `embedding`: 埋め込み関連
- `vector-search`: ベクトル検索関連
- `pipeline`: パイプライン・統合関連
- `reranking`: 再ランキング関連
- `llm`: 大規模言語モデル関連
- `generation`: テキスト生成関連
- `rag`: RAG（検索拡張生成）関連
- `ui`: ユーザーインターフェース関連
- `gradio`: Gradio フレームワーク関連
- `web`: Web関連
- `evaluation`: 評価・テスト関連
- `testing`: テスト関連
- `documentation`: ドキュメント関連
- `integration`: 統合関連

## 🔧 ワークフローのカスタマイズ

ワークフローファイル `.github/workflows/create-issues.yml` を編集することで、以下をカスタマイズできます：

- Issue タイトルのプレフィックス
- ラベルの追加・変更
- アサイニーの自動設定
- マイルストーンの自動設定
- Issue テンプレートの修正

## ⚠️ 注意事項

1. **権限**: ワークフロー実行にはリポジトリの `issues: write` 権限が必要です
2. **重複**: 同じタイトルのIssueが既に存在する場合も新しく作成されます
3. **レート制限**: GitHub APIのレート制限を避けるため、Issue作成間に1秒の遅延があります
4. **エラー処理**: 個別のIssue作成に失敗してもワークフローは継続します

## 📝 Issue フォーマット

各Issueは以下の形式で作成されます：

```markdown
> 📋 **このIssueは自動生成されました**
> - 予想作業時間が記載されています
> - チェックボックスで進捗を管理してください
> - 困った時は @RentoYabuki06 にメンション

---

[元のMarkdownファイルの内容がここに挿入されます]
```

## 🎯 推奨ワークフロー

1. **フェーズ1** (Issue #01-03): 基盤構築
2. **フェーズ2** (Issue #04-06): 検索システム構築  
3. **フェーズ3** (Issue #07-09): 応答生成システム
4. **フェーズ4** (Issue #10-12): UI・評価・品質保証

各フェーズを順番に進めることで、段階的にスキルアップしながらシステムを完成させることができます。