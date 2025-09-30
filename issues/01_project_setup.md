# Issue #01: プロジェクトの基本セットアップ

**予想時間**: 15分
**難易度**: 初級
**学習項目**: Python環境構築、依存関係管理

## 概要
Wikipedia RAGプロジェクトの基本的なディレクトリ構成とPython環境をセットアップする。

## タスク

### 1. ディレクトリ構造の作成
以下の構成でディレクトリを作成：
```
project/
  data/
  artifacts/
  src/
  requirements.txt
  README.md
```

### 2. requirements.txtの作成
以下のライブラリを含める：
```
transformers>=4.35.0
datasets>=2.14.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
accelerate>=0.24.0
gradio>=4.0.0
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
```

### 3. 基本的なREADME.mdの作成
- プロジェクトの概要
- インストール手順
- 基本的な使い方（後で更新予定）

### 4. .gitignoreの作成
- `artifacts/` ディレクトリ
- `__pycache__/`
- `.env`
- Python仮想環境関連ファイル

## 受け入れ基準
- [ ] ディレクトリ構造が正しく作成されている
- [ ] requirements.txtで`pip install -r requirements.txt`が成功する
- [ ] README.mdに基本情報が記載されている
- [ ] .gitignoreが適切に設定されている

## 学習ポイント
- Python依存関係管理の基本
- プロジェクト構成のベストプラクティス
- GitHubリポジトリの基本的な構成

## 次のステップ
Issue #02: Wikipediaデータ取得機能の実装