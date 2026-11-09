"""RAG 评测：召回命中率 / 元数据约束满足率 / 答案正确率 / 端到端延迟。

用法：
    python eval/run_eval.py                 # 读 eval/questions.jsonl
    python eval/run_eval.py --limit 5       # 只跑前 5 条（省 token）
    python eval/run_eval.py --no-filter     # 关闭元数据过滤，做对照实验

输出：逐条结果 + 汇总指标，同时写入 eval/report.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main import RecipeRAGSystem  # noqa: E402


def load_questions(path: Path) -> list:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-filter", action="store_true", help="关闭元数据过滤（对照组）")
    parser.add_argument("--post-filter", action="store_true", help="检索后再过滤（对照组，用于复现初版的 0.50）")
    parser.add_argument("--report", default=str(ROOT / "eval" / "report.json"))
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    if args.limit:
        questions = questions[: args.limit]

    rag = RecipeRAGSystem()
    if args.no_filter:
        rag.config.use_metadata_filter = False
    if args.post_filter:
        rag.config.filter_mode = "post"
    rag.initialize_system()
    rag.build_knowledge_base()

    # 捕获检索结果，便于计算召回与约束满足
    captured = {"docs": []}
    for name in ("hybrid_search", "metadata_filtered_search", "post_filtered_search"):
        original = getattr(rag.retrieval_module, name)

        def make_wrapper(func):
            def wrapper(*a, **kw):
                docs = func(*a, **kw)
                captured["docs"] = docs
                return docs
            return wrapper

        setattr(rag.retrieval_module, name, make_wrapper(original))

    rows = []
    for q in questions:
        captured["docs"] = []
        t0 = time.time()
        try:
            answer = rag.ask_question(q["question"], stream=False)
            error = None
        except Exception as exc:  # noqa: BLE001
            answer, error = "", f"{type(exc).__name__}: {exc}"
        latency = time.time() - t0

        docs = captured["docs"]
        # 检索返回的是 chunk：同一道菜的食材 / 步骤 / 小贴士是三个不同的块。
        # 用户看到的是 get_parent_documents() 按菜品去重后的结果，所以指标也必须
        # 按菜品算。此前直接对 chunk 取 dish_name，导致「金针菇汤、金针菇汤、
        # 金针菇汤」被算成三条有效推荐，列表型约束满足率虚高到 1.00。
        unique_docs = []
        seen_dishes: set[str] = set()
        for d in docs:
            name = d.metadata.get("dish_name", "")
            if name in seen_dishes:
                continue
            seen_dishes.add(name)
            unique_docs.append(d)

        retrieved = [d.metadata.get("dish_name", "") for d in unique_docs]
        retrieved_cat = [d.metadata.get("category", "") for d in unique_docs]
        retrieved_diff = [d.metadata.get("difficulty", "") for d in unique_docs]

        if q["type"] == "detail":
            retrieval_hit = q["expect_dish"] in retrieved[:3]
            answer_hit = q["expect_dish"] in (answer or "")
            category_ok = None
            difficulty_ok = None
        else:
            retrieval_hit = None
            answer_hit = None
            category_ok = bool(retrieved) and all(c == q["expect_category"] for c in retrieved_cat)
            difficulty_ok = bool(retrieved) and all(d == q["expect_difficulty"] for d in retrieved_diff)

        rows.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "answer": answer,
            "error": error,
            "retrieved_dishes": retrieved[:5],
            "distinct_dishes": len(retrieved),
            "retrieval_hit": retrieval_hit,
            "answer_hit": answer_hit,
            "category_ok": category_ok,
            "difficulty_ok": difficulty_ok,
            "empty": len(retrieved) == 0,
            "latency_s": round(latency, 2),
        })
        print(f"[{q['id']}|{q['type']}] hit={retrieval_hit} ans={answer_hit} "
              f"cat_ok={category_ok} diff_ok={difficulty_ok} empty={len(retrieved) == 0} "
              f"distinct={len(retrieved)} {latency:.1f}s :: {q['question']}", flush=True)

    detail_rows = [r for r in rows if r["type"] == "detail"]
    list_rows = [r for r in rows if r["type"] == "list"]
    errors = [r for r in rows if r["error"]]

    summary = {
        "mode": "no_filter" if args.no_filter else ("post_filter" if args.post_filter else "hybrid_filter"),
        "total": len(rows),
        "errors": len(errors),
        "detail_count": len(detail_rows),
        "detail_recall@3": round(
            sum(1 for r in detail_rows if r["retrieval_hit"]) / len(detail_rows), 3
        ) if detail_rows else None,
        "detail_answer_hit": round(
            sum(1 for r in detail_rows if r["answer_hit"]) / len(detail_rows), 3
        ) if detail_rows else None,
        "list_count": len(list_rows),
        "list_category_ok": round(
            sum(1 for r in list_rows if r["category_ok"]) / len(list_rows), 3
        ) if list_rows else None,
        "list_difficulty_ok": round(
            sum(1 for r in list_rows if r["difficulty_ok"]) / len(list_rows), 3
        ) if list_rows else None,
        # 列表型问题承诺「推荐 3 道菜」。约束满足率只看「返回的菜对不对」，
        # 不看「够不够数」——只推荐 1 道菜也能是 1.00。这两个指标补上那个缺口。
        "list_distinct_mean": round(
            statistics.mean(r["distinct_dishes"] for r in list_rows), 2
        ) if list_rows else None,
        "list_has_three": round(
            sum(1 for r in list_rows if r["distinct_dishes"] >= 3) / len(list_rows), 3
        ) if list_rows else None,
        "empty_result_rate": round(sum(1 for r in rows if r["empty"]) / len(rows), 3) if rows else None,
        "latency_mean_s": round(statistics.mean(r["latency_s"] for r in rows), 2) if rows else None,
        "latency_p90_s": round(
            statistics.quantiles([r["latency_s"] for r in rows], n=10)[8], 2
        ) if len(rows) >= 10 else None,
    }

    print("\n=== 汇总 ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    Path(args.report).write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n明细已写入 {args.report}")


if __name__ == "__main__":
    main()
