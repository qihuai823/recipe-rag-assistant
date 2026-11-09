"""
检索优化模块
"""

import logging
import hashlib
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class RetrievalOptimizationModule:
    """检索优化模块 - 负责混合检索和过滤"""
    
    def __init__(self, vectorstore: FAISS, chunks: List[Document]):
        """
        初始化检索优化模块
        
        Args:
            vectorstore: FAISS向量存储
            chunks: 文档块列表
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.setup_retrievers()

    def setup_retrievers(self):
        """设置向量检索器和BM25检索器"""
        logger.info("正在设置检索器...")

        # 候选池大小。实际返回条数由 hybrid_search / metadata_filtered_search 的
        # top_k 参数控制：列表型问题需要 12 个块才能在按菜品去重后凑够 3 道菜，
        # 而检索器若固定取 5 条，上面传多大的 top_k 都拿不到更多候选。
        pool = 20

        # 向量检索器
        self.vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": pool}
        )

        # BM25检索器
        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=pool
        )

        logger.info(f"检索器设置完成（候选池 {pool}）")
    
    def hybrid_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        混合检索 - 结合向量检索和BM25检索，使用RRF重排

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            检索到的文档列表
        """
        # 分别获取向量检索和BM25检索结果
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # 使用RRF重排
        reranked_docs = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked_docs[:top_k]
    
    def metadata_filtered_search(self, query: str, filters: Dict[str, Any], top_k: int = 5) -> List[Document]:
        """检索前过滤：先在元数据上筛出子集，再在子集内做向量 + BM25 混合检索。

        与「先取 top-k 再过滤」相比，不会因为候选池里恰好没有符合条件的文档
        而误判为「无结果」。
        """
        candidates = [doc for doc in self.chunks if self._match(doc, filters)]
        if not candidates:
            logger.info(f"过滤条件 {filters} 在语料中没有匹配文档")
            return []

        # 向量检索：把过滤条件下推给 FAISS
        try:
            vector_docs = self.vectorstore.similarity_search(query, k=top_k * 2, filter=filters or None)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"FAISS 带过滤检索失败({exc})，退化为检索后过滤")
            vector_docs = [
                d for d in self.vectorstore.similarity_search(query, k=top_k * 3) if self._match(d, filters)
            ][: top_k]

        # BM25：只在过滤后的子集上建索引
        bm25_docs = BM25Retriever.from_documents(candidates, k=top_k).invoke(query)

        reranked = self._rrf_rerank(vector_docs, bm25_docs)
        return reranked[:top_k]

    def post_filtered_search(
        self, query: str, filters: Dict[str, Any], top_k: int = 3, pool: int = 5
    ) -> List[Document]:
        """检索后过滤（对照组）：先取 top-k 候选，再按元数据筛。

        这是项目的初版做法，也是难度条件满足率只有 0.50 的原因：候选池只有
        pool 条，如果里面恰好没有目标难度的文档，就会被误判成「无结果」，
        于是兜底逻辑放宽难度，返回的菜和用户要求的不符。

        保留这个「已知更差」的实现不是留垃圾，是为了让 README 里那个 0.50
        可以复现——否则它只是文档里一句没有证据的话，面试时经不起追问。
        """
        candidates = self.hybrid_search(query, top_k=pool)
        return [doc for doc in candidates if self._match(doc, filters)][:top_k]

    @staticmethod
    def _match(doc: Document, filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if key not in doc.metadata:
                return False
            if isinstance(value, list):
                if doc.metadata[key] not in value:
                    return False
            elif doc.metadata[key] != value:
                return False
        return True

    def _rrf_rerank(self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """
        使用RRF (Reciprocal Rank Fusion) 算法重排文档

        Args:
            vector_docs: 向量检索结果
            bm25_docs: BM25检索结果
            k: RRF参数，用于平滑排名

        Returns:
            重排后的文档列表
        """
        doc_scores = {}
        doc_objects = {}

        # 计算向量检索结果的RRF分数
        for rank, doc in enumerate(vector_docs):
            # 使用文档内容的确定性哈希作为唯一标识
            doc_id = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            doc_objects[doc_id] = doc

            # RRF公式: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"向量检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 计算BM25检索结果的RRF分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            doc_objects[doc_id] = doc

            rrf_score = 1.0 / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

            logger.debug(f"BM25检索 - 文档{rank+1}: RRF分数 = {rrf_score:.4f}")

        # 按最终RRF分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                # 将RRF分数添加到文档元数据中
                doc.metadata['rrf_score'] = final_score
                reranked_docs.append(doc)
                logger.debug(f"最终排序 - 文档: {doc.page_content[:50]}... 最终RRF分数: {final_score:.4f}")

        logger.info(f"RRF重排完成: 向量检索{len(vector_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档")

        return reranked_docs
