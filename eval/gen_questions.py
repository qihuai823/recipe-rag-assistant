"""从菜谱数据生成评测集（不需要调用大模型）。

用法：
    python eval/gen_questions.py                      # 默认 45 条详情 + 20 条列表
    python eval/gen_questions.py --detail 60 --list 30
    python eval/gen_questions.py --seed 7

为什么要能调规模：21 条的时候单条波动就能左右整体通过率，看不出策略改动的
真实影响。扩到几十条之后指标才有统计意义——但也别贪多，跑一轮要真金白银的
时间和 token。
"""

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cook" / "dishes"
OUT = ROOT / "eval" / "questions.jsonl"

CATEGORY_LABELS = {
    "meat_dish": "荤菜",
    "vegetable_dish": "素菜",
    "soup": "汤品",
    "dessert": "甜品",
    "breakfast": "早餐",
    "staple": "主食",
    "aquatic": "水产",
    "condiment": "调料",
    "drink": "饮品",
}
DIFFICULTY_BY_STARS = {1: "非常简单", 2: "简单", 3: "中等", 4: "困难", 5: "非常困难"}

# 列表类问题只在这几个分类里出：调料/饮品在数据里样本太少，
# 问「有什么简单的饮品」得到的候选不足，指标会被数据量而不是检索质量左右。
LIST_CATEGORIES = ("荤菜", "素菜", "汤品", "甜品", "主食", "早餐", "水产")


def parse(path: Path, category_dir: str) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    stars = len(re.findall(r"[★]", lines[2] if len(lines) > 2 else ""))
    return {
        "dish": path.stem,
        "category": CATEGORY_LABELS.get(category_dir, category_dir),
        "difficulty": DIFFICULTY_BY_STARS.get(stars, "中等"),
    }


def load_dishes() -> list[dict]:
    dishes = []
    for category_dir in CATEGORY_LABELS:
        folder = DATA / category_dir
        if not folder.is_dir():
            continue
        for md in folder.glob("*.md"):
            dishes.append(parse(md, category_dir))
    return dishes


def build_list_combos(dishes: list[dict], limit: int, rng: random.Random) -> list[tuple[str, str, str]]:
    """挑样本足够的 (分类, 难度) 组合，避免用稀薄数据构造必然失败的问题。"""
    combos = []
    for category in LIST_CATEGORIES:
        for difficulty in DIFFICULTY_BY_STARS.values():
            pool = [d["dish"] for d in dishes if d["category"] == category and d["difficulty"] == difficulty]
            if len(pool) >= 2:
                combos.append((category, difficulty, rng.choice(pool), len(pool)))
    # 候选多的组合优先，让列表类问题落在检索确实有得挑的场景上
    combos.sort(key=lambda item: item[3], reverse=True)
    picked = combos[:limit]
    rng.shuffle(picked)
    return [(category, difficulty, sample) for category, difficulty, sample, _ in picked]


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 RAG 评测集")
    parser.add_argument("--detail", type=int, default=45, help="详情类问题条数")
    parser.add_argument("--list", type=int, default=20, help="列表类问题条数")
    parser.add_argument("--seed", type=int, default=20261029, help="采样随机种子（保证可复现）")
    parser.add_argument("--out", default=str(OUT), help="输出文件路径")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dishes = load_dishes()
    if not dishes:
        raise SystemExit(f"没有找到菜谱数据：{DATA}")

    detail_pool = [d for d in dishes if d["category"] in {"荤菜", "素菜", "汤品", "主食", "水产"}]
    detail = rng.sample(detail_pool, min(args.detail, len(detail_pool)))
    listing = build_list_combos(dishes, args.list, rng)

    rows = []
    for index, dish in enumerate(detail, start=1):
        rows.append(
            {
                "id": f"D{index:02d}",
                "type": "detail",
                "question": f"{dish['dish']}怎么做？",
                "expect_dish": dish["dish"],
                "expect_category": dish["category"],
                "expect_difficulty": dish["difficulty"],
            }
        )
    for index, (category, difficulty, sample) in enumerate(listing, start=1):
        rows.append(
            {
                "id": f"L{index:02d}",
                "type": "list",
                "question": f"有什么{difficulty}的{category}推荐？",
                "expect_dish": None,
                "expect_category": category,
                "expect_difficulty": difficulty,
                "sample_dish": sample,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"菜谱总数 {len(dishes)}，写入 {len(rows)} 条评测问题（详情 {len(detail)} / 列表 {len(listing)}）-> {out}")


if __name__ == "__main__":
    main()
