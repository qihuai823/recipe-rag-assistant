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
    use_metadata_filter: bool = os.getenv("USE_METADATA_FILTER", "1") == "1"

    # 生成配置
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "2048"))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = RAGConfig()
