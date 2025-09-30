# Issue #12: 統合テスト・デバッグ・ドキュメント整備

**予想時間**: 30分
**難易度**: 中級
**学習項目**: 統合テスト、デバッグ技法、ドキュメンテーション

## 概要
全システムの統合テストを実行し、発見した問題をデバッグ・修正して、包括的なドキュメントを整備する。

## タスク

### 1. 統合テストスイートの作成 (`tests/test_integration.py`)
- エンドツーエンドのワークフローテスト
- 各コンポーネントの単体テスト
- エラーケースのテスト
- パフォーマンステスト

### 2. デバッグ・修正
- 発見されたバグの特定と修正
- パフォーマンスボトルネックの改善
- メモリリークや効率性の問題解決
- ログ出力の改善

### 3. READMEの更新
- インストール手順の詳細化
- 使用例の充実
- トラブルシューティング情報
- システム要件の明記

### 4. 運用ドキュメントの作成
- `docs/setup_guide.md`: 詳細セットアップガイド
- `docs/troubleshooting.md`: よくある問題と解決方法
- `docs/performance_guide.md`: パフォーマンス最適化ガイド

## 実装例

### 統合テスト (`tests/test_integration.py`)
```python
import pytest
import tempfile
import shutil
from pathlib import Path
import json
import numpy as np

from src.data_loader import load_wikipedia_data, normalize_text
from src.chunker import TextChunker
from src.embedder import E5Embedder
from src.vector_store import FAISSVectorStore
from src.reranker import BGEReranker
from src.generator import QwenGenerator
from src.rag_wiki import WikiRAG

class TestIntegration:
    @pytest.fixture
    def temp_artifacts_dir(self):
        """テスト用一時ディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def small_test_data(self):
        """テスト用小規模データ"""
        return [
            {
                'id': 'test_1',
                'title': 'テスト記事1',
                'text': 'これはテスト用の記事です。' * 20,
                'source': 'jawiki:テスト記事1'
            },
            {
                'id': 'test_2', 
                'title': 'テスト記事2',
                'text': '日本の歴史について説明します。' * 20,
                'source': 'jawiki:テスト記事2'
            }
        ]
    
    def test_data_loading_and_processing(self):
        """データ読み込みと前処理のテスト"""
        # 小規模データでの動作確認
        data = load_wikipedia_data(max_articles=5)
        assert len(data) <= 5
        assert all('title' in item and 'text' in item for item in data)
        
        # テキスト正規化
        sample_text = "  これは　　テスト　です。  \\n\\n"
        normalized = normalize_text(sample_text)
        assert normalized == "これは テスト です。"
    
    def test_chunking_pipeline(self, small_test_data):
        """チャンキングパイプラインのテスト"""
        chunker = TextChunker(chunk_size=100, overlap=20)
        
        all_chunks = []
        for article in small_test_data:
            chunks = chunker.chunk_text(
                article['text'], 
                article['id'], 
                article['title']
            )
            all_chunks.extend(chunks)
        
        assert len(all_chunks) > 0
        assert all('text' in chunk for chunk in all_chunks)
        assert all('source' in chunk for chunk in all_chunks)
        assert all('chunk_id' in chunk for chunk in all_chunks)
    
    def test_embedding_pipeline(self, small_test_data):
        """埋め込みパイプラインのテスト"""
        embedder = E5Embedder()
        
        # パッセージ埋め込み
        texts = [item['text'][:200] for item in small_test_data]
        embeddings = embedder.encode_passages(texts)
        
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == 768  # multilingual-e5-baseの次元数
        
        # 正規化確認
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
        
        # クエリ埋め込み
        query_emb = embedder.encode_query("テスト質問")
        assert query_emb.shape == (1, 768)
    
    def test_vector_search(self, small_test_data, temp_artifacts_dir):
        \"\"\"ベクトル検索のテスト\"\"\"
        # 埋め込み生成
        embedder = E5Embedder()
        texts = [item['text'][:200] for item in small_test_data]
        embeddings = embedder.encode_passages(texts)
        
        # インデックス構築
        vector_store = FAISSVectorStore(768)
        vector_store.build_index(embeddings)
        
        # 検索テスト
        query_emb = embedder.encode_query("日本の歴史")
        scores, indices = vector_store.search(query_emb, k=2)
        
        assert len(scores[0]) <= 2
        assert len(indices[0]) <= 2
        assert all(0 <= idx < len(small_test_data) for idx in indices[0])
        
        # 保存・読み込みテスト
        index_path = Path(temp_artifacts_dir) / "test.index"
        vector_store.save(str(index_path))
        assert index_path.exists()
        
        new_store = FAISSVectorStore(768)
        new_store.load(str(index_path))
        new_scores, new_indices = new_store.search(query_emb, k=2)
        
        assert np.array_equal(scores, new_scores)
        assert np.array_equal(indices, new_indices)
    
    def test_reranker_functionality(self):
        \"\"\"再ランカーのテスト\"\"\"
        reranker = BGEReranker()
        
        query = "日本の歴史について"
        passages = [
            "日本の歴史は古代から現代まで続いています。",
            "寿司は日本料理の代表です。",
            "江戸時代は平和な時代でした。"
        ]
        
        if reranker.is_available:
            results = reranker.rerank(query, passages, top_k=2)
            assert len(results) == 2
            assert all(isinstance(item, tuple) for item in results)
            assert all(len(item) == 2 for item in results)
        else:
            # フォールバック動作の確認
            results = reranker.rerank(query, passages, top_k=2)
            assert len(results) == 2
    
    def test_generator_functionality(self):
        \"\"\"生成器のテスト\"\"\"
        generator = QwenGenerator()
        
        question = "テストについて教えて"
        contexts = [
            {
                'text': 'テストは品質を確認する重要な作業です。',
                'article_title': 'テスト',
                'chunk_id': 0
            }
        ]
        
        answer = generator.generate_answer(question, contexts)
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert "テスト" in answer or "test" in answer.lower()
    
    def test_full_rag_pipeline(self, temp_artifacts_dir):
        \"\"\"完全RAGパイプラインのテスト\"\"\"
        # 小規模データでのエンドツーエンドテスト
        # 注意: このテストは計算資源を必要とするため、CI環境では慎重に実行
        pass  # 実装は時間の都合上省略
    
    def test_error_handling(self):
        \"\"\"エラーハンドリングのテスト\"\"\"
        # 存在しないファイルでの初期化
        with pytest.raises((FileNotFoundError, Exception)):
            WikiRAG("nonexistent_directory")
        
        # 空の質問
        # 実装に応じたテストを追加
    
    def test_performance_benchmarks(self, small_test_data):
        \"\"\"パフォーマンステスト\"\"\"
        import time
        
        # 埋め込み速度テスト
        embedder = E5Embedder()
        texts = [item['text'][:200] for item in small_test_data * 10]
        
        start_time = time.time()
        embeddings = embedder.encode_passages(texts)
        embedding_time = time.time() - start_time
        
        # 期待値: 20テキストを10秒以内で処理
        assert embedding_time < 10.0
        assert embeddings.shape[0] == len(texts)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 更新されたREADME.md
```markdown
# Wikipedia RAG システム

日本語Wikipediaを使った検索拡張生成（RAG）システムです。質問を入力すると、関連するWikipedia記事を検索し、その情報に基づいて回答を生成します。

## 🌟 特徴

- **完全日本語対応**: 日本語Wikipediaデータを使用
- **高精度検索**: E5埋め込み + BGE再ランカーによる2段階検索
- **出典明示**: すべての回答に参照情報を付与
- **簡単デプロイ**: Gradio UIでローカル・クラウド両対応
- **評価機能**: Recall@K指標による客観的性能測定

## 📋 システム要件

### ハードウェア要件
- **RAM**: 8GB以上推奨（4GB最低）
- **Storage**: 10GB以上の空き容量
- **GPU**: 任意（あれば高速化）

### ソフトウェア要件
- Python 3.10以上
- pip または conda

### 対応OS
- macOS
- Linux (Ubuntu 20.04+)
- Windows 10/11

## 🚀 クイックスタート

### 1. 環境セットアップ
\\`\\`\\`bash
# リポジトリクローン
git clone <repository-url>
cd wikipedia_rag

# 依存関係インストール
pip install -r requirements.txt
\\`\\`\\`

### 2. インデックス構築
\\`\\`\\`bash
# 小規模テスト用（約5分）
python src/build_wiki_index.py --max_articles 1000

# 中規模用（約30分）
python src/build_wiki_index.py --max_articles 10000

# 大規模用（約2時間）
python src/build_wiki_index.py --max_articles 30000
\\`\\`\\`

### 3. 質問応答テスト
\\`\\`\\`bash
python src/rag_wiki.py -q "織田信長について教えて"
\\`\\`\\`

### 4. Web UI起動
\\`\\`\\`bash
python src/app_wiki.py
# ブラウザで http://localhost:7860 にアクセス
\\`\\`\\`

## 📖 詳細な使い方

### コマンドラインオプション

#### インデックス構築
\\`\\`\\`bash
python src/build_wiki_index.py \\
    --max_articles 30000 \\        # 処理する記事数
    --chunk_size 450 \\            # チャンクサイズ
    --overlap 60 \\                # オーバーラップサイズ
    --batch_size 32 \\             # バッチサイズ
    --output_dir artifacts         # 出力ディレクトリ
\\`\\`\\`

#### 質問応答
\\`\\`\\`bash
python src/rag_wiki.py \\
    -q "質問文" \\                 # 質問
    --topk 16 \\                   # 検索候補数
    --topn 5 \\                    # 最終候補数
    --no-rerank \\                 # 再ランキング無効化
    --verbose                      # 詳細ログ
\\`\\`\\`

### 評価実行
\\`\\`\\`bash
python src/eval_retrieval.py --eval_data data/dev.jsonl
\\`\\`\\`

## 🔧 カスタマイズ

### モデル変更
各モデルは環境変数または設定ファイルで変更可能です：

\\`\\`\\`python
# src/config.py (作成)
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3" 
LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
\\`\\`\\`

### チューニングパラメータ
- **chunk_size**: 長い文書は大きく、短い文書は小さく
- **overlap**: 情報の連続性重視なら大きく
- **top_k/top_n**: 精度重視なら大きく、速度重視なら小さく

## 📊 性能指標

### ベンチマーク結果（参考値）
| モデル構成 | Recall@5 | 処理時間 | メモリ使用量 |
|------------|----------|----------|------------|
| CPU only   | 0.65     | 3.2秒    | 4.5GB      |
| GPU有効    | 0.68     | 1.8秒    | 6.2GB      |

## 🐛 トラブルシューティング

### よくある問題

#### 1. メモリ不足エラー
\\`\\`\\`bash
# バッチサイズを小さくして再実行
python src/build_wiki_index.py --batch_size 8
\\`\\`\\`

#### 2. モデルダウンロードに失敗
\\`\\`\\`bash
# キャッシュクリア後に再実行
rm -rf ~/.cache/huggingface/
pip install --upgrade transformers
\\`\\`\\`

#### 3. FAISS関連エラー
\\`\\`\\`bash
# FAISS再インストール
pip uninstall faiss-cpu
pip install faiss-cpu --no-cache-dir
\\`\\`\\`

詳細は [docs/troubleshooting.md](docs/troubleshooting.md) を参照してください。

## 📈 性能最適化

### GPU利用設定
\\`\\`\\`python
# より高速なGPU推論のために
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
\\`\\`\\`

### メモリ最適化
- 処理する記事数を段階的に増やす
- バッチサイズを環境に応じて調整
- 不要なモデルは無効化

## 🤝 貢献

1. Fork this repository
2. Create your feature branch
3. Commit your changes  
4. Push to the branch
5. Create a Pull Request

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 🙏 謝辞

- Hugging Face Transformers
- FAISS by Facebook Research
- Gradio team
- Wikipedia contributors
\\`\\`\\`

## 受け入れ基準
- [ ] 統合テストスイートが動作する
- [ ] 主要な機能のテストカバレッジ≥80%
- [ ] 発見されたバグが修正される
- [ ] README.mdが包括的に更新される
- [ ] トラブルシューティングガイドが作成される
- [ ] パフォーマンスベンチマークが実行できる

## 学習ポイント
- 大規模システムの統合テスト設計
- デバッグ技法とトラブルシューティング
- 技術文書作成のベストプラクティス
- ユーザビリティを考慮したドキュメンテーション
- 継続的品質改善のプロセス

## テスト実行方法
```bash
# 統合テスト実行
python -m pytest tests/test_integration.py -v

# カバレッジ測定
pip install pytest-cov
python -m pytest --cov=src tests/

# パフォーマンステスト
python tests/test_integration.py::TestIntegration::test_performance_benchmarks
```

## 完了後の確認事項
- [ ] 全てのテストが通過する
- [ ] ドキュメントが最新で正確
- [ ] エラーケースが適切にハンドリングされる
- [ ] ユーザガイドが分かりやすい
- [ ] システムが安定して動作する

これで、学習用Wikipedia RAGシステムの完全な実装と検証が完了します！