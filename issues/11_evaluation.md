# Issue #11: 評価データセットの作成とRecall@K評価機能の実装

**予想時間**: 25分
**難易度**: 中級
**学習項目**: 情報検索評価、Recall@K、評価データセット設計

## 概要
RAGシステムの検索性能を客観的に評価するための評価データセットを作成し、Recall@K指標を計算する機能を実装する。

## タスク

### 1. 評価データセットの作成 (`data/dev.jsonl`)
- 日本語の質問と正解ソースのペア作成
- Wikipedia記事に基づいた実在する情報での質問設計
- 多様な質問タイプ（人物、出来事、概念など）の包括

### 2. 評価スクリプトの実装 (`src/eval_retrieval.py`)
- Recall@K計算機能
- 複数の評価指標（Recall@1, @3, @5）
- 詳細な評価レポート生成

### 3. 評価データ形式の定義
```json
{
    "question": "質問文",
    "gold_sources": ["jawiki:記事タイトル#chunk=番号", ...],
    "category": "人物|出来事|概念|その他",
    "difficulty": "easy|medium|hard"
}
```

### 4. 自動評価パイプライン
- 評価データセット全体での性能測定
- カテゴリ別・難易度別の分析
- 失敗ケースの詳細分析

## 実装例のスケルトン

### 評価データセットサンプル (`data/dev.jsonl`)
```jsonl
{"question": "織田信長が本能寺の変で亡くなったのは何年ですか？", "gold_sources": ["jawiki:織田信長#chunk=5", "jawiki:本能寺の変#chunk=1"], "category": "人物", "difficulty": "easy"}
{"question": "大政奉還が行われた具体的な日付を教えてください", "gold_sources": ["jawiki:大政奉還#chunk=1"], "category": "出来事", "difficulty": "medium"}
{"question": "江戸幕府の成立から終わりまでの期間はどのくらいですか？", "gold_sources": ["jawiki:江戸幕府#chunk=1", "jawiki:江戸時代#chunk=0"], "category": "概念", "difficulty": "medium"}
{"question": "坂本龍馬が薩長同盟の仲介で果たした具体的な役割について", "gold_sources": ["jawiki:坂本龍馬#chunk=3", "jawiki:薩長同盟#chunk=2"], "category": "人物", "difficulty": "hard"}
{"question": "明治維新における廃藩置県の目的と効果を説明してください", "gold_sources": ["jawiki:廃藩置県#chunk=0", "jawiki:明治維新#chunk=4"], "category": "出来事", "difficulty": "hard"}
```

### 評価スクリプト (`src/eval_retrieval.py`)
```python
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from rag_wiki import WikiRAG

class RetrievalEvaluator:
    def __init__(self, rag_system: WikiRAG):
        """検索評価器を初期化"""
        self.rag_system = rag_system
        
    def load_evaluation_data(self, filepath: str) -> List[Dict[str, Any]]:
        """評価データを読み込み"""
        eval_data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                eval_data.append(json.loads(line.strip()))
        return eval_data
    
    def evaluate_single_question(self, question: str, gold_sources: List[str], 
                               top_k: int = 16) -> Dict[str, Any]:
        """単一質問の評価"""
        try:
            # 検索実行（生成は行わず、検索結果のみ取得）
            query_embedding = self.rag_system.embedder.encode_query(question)
            scores, indices = self.rag_system.vector_store.search(query_embedding, k=top_k)
            
            # 検索結果の取得
            retrieved_sources = []
            for idx in indices[0]:
                if idx < len(self.rag_system.metadata):
                    metadata = self.rag_system.metadata[idx]
                    source = f"jawiki:{metadata['article_title']}#chunk={metadata['chunk_id']}"
                    retrieved_sources.append(source)
            
            # Recall@K計算
            recall_metrics = {}
            for k in [1, 3, 5, 10]:
                if k <= len(retrieved_sources):
                    top_k_retrieved = set(retrieved_sources[:k])
                    gold_set = set(gold_sources)
                    recall = len(top_k_retrieved.intersection(gold_set)) / len(gold_set) if gold_set else 0.0
                    recall_metrics[f'recall@{k}'] = recall
                else:
                    recall_metrics[f'recall@{k}'] = 0.0
            
            return {
                'retrieved_sources': retrieved_sources,
                'gold_sources': gold_sources,
                'metrics': recall_metrics,
                'success': True
            }
            
        except Exception as e:
            logging.error(f"Evaluation error for question: {question[:50]}... - {e}")
            return {
                'retrieved_sources': [],
                'gold_sources': gold_sources,
                'metrics': {f'recall@{k}': 0.0 for k in [1, 3, 5, 10]},
                'success': False,
                'error': str(e)
            }
    
    def evaluate_dataset(self, eval_data: List[Dict[str, Any]], top_k: int = 16) -> Dict[str, Any]:
        """データセット全体の評価"""
        all_metrics = defaultdict(list)
        category_metrics = defaultdict(lambda: defaultdict(list))
        difficulty_metrics = defaultdict(lambda: defaultdict(list))
        failed_questions = []
        
        logging.info(f"Evaluating {len(eval_data)} questions...")
        
        for i, item in enumerate(eval_data):
            if (i + 1) % 10 == 0:
                logging.info(f"Processed {i + 1}/{len(eval_data)} questions")
            
            question = item['question']
            gold_sources = item['gold_sources']
            category = item.get('category', 'unknown')
            difficulty = item.get('difficulty', 'unknown')
            
            result = self.evaluate_single_question(question, gold_sources, top_k)
            
            if result['success']:
                # 全体メトリクス
                for metric_name, value in result['metrics'].items():
                    all_metrics[metric_name].append(value)
                
                # カテゴリ別メトリクス
                for metric_name, value in result['metrics'].items():
                    category_metrics[category][metric_name].append(value)
                
                # 難易度別メトリクス
                for metric_name, value in result['metrics'].items():
                    difficulty_metrics[difficulty][metric_name].append(value)
            else:
                failed_questions.append({
                    'question': question,
                    'error': result.get('error', 'Unknown error')
                })
        
        # 平均値計算
        def calculate_averages(metrics_dict):
            return {k: sum(v) / len(v) if v else 0.0 for k, v in metrics_dict.items()}
        
        results = {
            'overall_metrics': calculate_averages(all_metrics),
            'category_metrics': {cat: calculate_averages(metrics) 
                               for cat, metrics in category_metrics.items()},
            'difficulty_metrics': {diff: calculate_averages(metrics) 
                                 for diff, metrics in difficulty_metrics.items()},
            'total_questions': len(eval_data),
            'successful_evaluations': len(eval_data) - len(failed_questions),
            'failed_questions': failed_questions
        }
        
        return results
    
    def print_evaluation_report(self, results: Dict[str, Any]):
        """評価結果レポートを出力"""
        print("\\n" + "=" * 60)
        print("RAG システム 検索性能評価レポート")
        print("=" * 60)
        
        # 全体結果
        print(f"\\n📊 全体結果 ({results['successful_evaluations']}/{results['total_questions']} 件成功)")
        for metric, value in results['overall_metrics'].items():
            print(f"  {metric.upper()}: {value:.3f}")
        
        # カテゴリ別結果
        if results['category_metrics']:
            print(f"\\n📂 カテゴリ別結果:")
            for category, metrics in results['category_metrics'].items():
                print(f"  {category}:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value:.3f}")
        
        # 難易度別結果
        if results['difficulty_metrics']:
            print(f"\\n⭐ 難易度別結果:")
            for difficulty, metrics in results['difficulty_metrics'].items():
                print(f"  {difficulty}:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value:.3f}")
        
        # 失敗ケース
        if results['failed_questions']:
            print(f"\\n❌ 失敗ケース ({len(results['failed_questions'])} 件):")
            for i, failed in enumerate(results['failed_questions'][:5]):  # 最大5件表示
                print(f"  {i+1}. {failed['question'][:50]}...")
                print(f"     エラー: {failed['error']}")
        
        print("=" * 60)

def parse_arguments():
    parser = argparse.ArgumentParser(description="RAG検索性能評価")
    parser.add_argument("--eval_data", default="data/dev.jsonl", help="評価データファイル")
    parser.add_argument("--artifacts_dir", default="artifacts", help="アーティファクトディレクトリ")
    parser.add_argument("--top_k", type=int, default=16, help="検索時のtop-k")
    parser.add_argument("--output", help="結果をJSONで保存するファイル名")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # RAGシステム初期化
        logging.info("Initializing RAG system...")
        rag_system = WikiRAG(args.artifacts_dir)
        
        # 評価器初期化
        evaluator = RetrievalEvaluator(rag_system)
        
        # 評価データ読み込み
        if not Path(args.eval_data).exists():
            logging.error(f"Evaluation data file not found: {args.eval_data}")
            return
        
        eval_data = evaluator.load_evaluation_data(args.eval_data)
        logging.info(f"Loaded {len(eval_data)} evaluation questions")
        
        # 評価実行
        results = evaluator.evaluate_dataset(eval_data, args.top_k)
        
        # レポート出力
        evaluator.print_evaluation_report(results)
        
        # JSON保存（オプション）
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logging.info(f"Results saved to {args.output}")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
```

## 受け入れ基準
- [ ] 評価データセット（20問以上）が作成される
- [ ] Recall@K計算が正確に動作する
- [ ] カテゴリ別・難易度別の分析が機能する
- [ ] 評価レポートが見やすい形式で出力される
- [ ] 失敗ケースが適切に記録・表示される
- [ ] JSONでの結果保存機能が動作する

## 学習ポイント
- 情報検索における評価指標の理解
- Recall@Kの計算方法と意味
- 評価データセット設計のベストプラクティス
- システム性能の客観的測定方法
- 評価結果の効果的な可視化

## テスト方法
```bash
# 評価実行
python src/eval_retrieval.py --eval_data data/dev.jsonl

# 結果をファイルに保存
python src/eval_retrieval.py --output evaluation_results.json

# カスタム設定での評価
python src/eval_retrieval.py --top_k 20 --artifacts_dir artifacts
```

## 期待される結果
- Recall@5 ≥ 0.6（暫定基準）
- カテゴリ間での性能差の把握
- 改善が必要な領域の特定

## 次のステップ
Issue #12: 統合テスト・デバッグ・ドキュメント整備