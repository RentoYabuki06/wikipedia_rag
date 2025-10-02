"""
RAGシステムの設定ファイル
モデルやパラメータの設定を一元管理
"""

# モデル設定
DEFAULT_GENERATOR_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_EMBEDDER_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# 利用可能なモデル:
# - Qwen/Qwen2.5-0.5B-Instruct : 超高速、軽量
# - Qwen/Qwen2.5-1B-Instruct   : 高速、バランス型（推奨）
# - Qwen/Qwen2.5-1.5B-Instruct : やや高速、高品質
# - Qwen/Qwen2.5-3B-Instruct   : 低速、最高品質

# 生成パラメータ
GENERATION_MAX_NEW_TOKENS = 200
GENERATION_DO_SAMPLE = False

# 検索パラメータ
DEFAULT_TOP_K = 5  # ベクトル検索で取得する候補数（16→5で高速化）
DEFAULT_TOP_N = 3  # 最終的に使用する候補数（5→3で高速化）
USE_RERANKER = False  # 再ランカーをデフォルトで無効化（大幅に高速化）

# チャンキングパラメータ
CHUNK_SIZE = 450
CHUNK_OVERLAP = 60

# プロンプト設定
MAX_CONTEXTS_FOR_PROMPT = 2  # プロンプトに含めるコンテキストの最大数
MAX_CONTEXT_LENGTH = 200  # 各コンテキストの最大文字数

# パス設定
DEFAULT_ARTIFACTS_DIR = "artifacts"
