"""
RAG 系统配置

所有可变项都可以用环境变量覆盖，方便在不同机器 / 演示环境里跑，不用改代码。
"""

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class RAGConfig:
    """RAG 系统配置"""

    # 路径配置
    data_path: str = os.getenv("DATA_PATH", "./data/cook")
    index_save_path: str = os.getenv("INDEX_PATH", "./vector_index")

    # 模型配置
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    llm_model: str = os.getenv("LLM_MODEL", "deepseek-chat")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")

    # 检索配置
    top_k: int = int(os.getenv("TOP_K", "3"))
    # 列表型问题承诺「推荐 3 道菜」，但检索返回的是 chunk：同一道菜的食材/步骤/
    # 小贴士是三个块，回溯父文档去重后常常只剩 1~2 道菜。所以列表型要更大的候选池，
    # 去重之后才够 top_k 道不同的菜。详情型不需要——它只要命中那一道。
    list_candidate_k: int = int(os.getenv("LIST_CANDIDATE_K", "12"))
    use_metadata_filter: bool = os.getenv("USE_METADATA_FILTER", "1") == "1"
    # 过滤发生在哪一层：
    #   pre  = 先在元数据上筛子集，再在子集内检索（默认，效果最好）
    #   post = 先取 top-k 候选再筛（初版做法，候选池不够时会误判成无结果）
    # 保留 post 是为了让「为什么过滤必须前移」这个结论可复现，而不是只写在 README 里。
    filter_mode: str = os.getenv("FILTER_MODE", "pre")

    # 生成配置
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "2048"))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = RAGConfig()
