"""从菜谱数据生成评测集（不需要调用大模型）。

用法：
    python eval/gen_questions.py            # 生成 eval/questions.jsonl
"""

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


def parse(path: Path, category_dir: str) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    stars = len(re.findall(r"[★]", text.splitlines()[2] if len(text.splitlines()) > 2 else ""))
    return {
        "dish": path.stem,
        "category": CATEGORY_LABELS.get(category_dir, category_dir),
        "difficulty": DIFFICULTY_BY_STARS.get(stars, "中等"),
    }


def main() -> None:
    rng = random.Random(20261029)
    dishes = []
    for category_dir in CATEGORY_LABELS:
        folder = DATA / category_dir
        if not folder.is_dir():
            continue
        for md in folder.glob("*.md"):
            dishes.append(parse(md, category_dir))

    # 详情类：问具体某道菜怎么做
    detail_pool = [d for d in dishes if d["category"] in {"荤菜", "素菜", "汤品", "主食", "水产"}]
    detail = rng.sample(detail_pool, 15)

    # 列表类：按 分类 + 难度 组合筛选
    combos = [("素菜", "简单"), ("荤菜", "中等"), ("汤品", "简单"),
              ("甜品", "简单"), ("主食", "中等"), ("早餐", "非常简单"),
              ("水产", "中等"), ("素菜", "非常简单")]
    listing = []
    for category, difficulty in combos:
        pool = [d["dish"] for d in dishes if d["category"] == category and d["difficulty"] == difficulty]
        if len(pool) >= 2:
            listing.append((category, difficulty, rng.choice(pool)))

    rows = []
    for d in detail:
        rows.append({
            "id": f"D{len(rows)+1:02d}",
            "type": "detail",
            "question": f"{d['dish']}怎么做？",
            "expect_dish": d["dish"],
            "expect_category": d["category"],
            "expect_difficulty": d["difficulty"],
        })
    for category, difficulty, sample in listing:
        rows.append({
            "id": f"L{len(rows)-len(detail)+1:02d}",
            "type": "list",
            "question": f"有什么{difficulty}的{category}推荐？",
            "expect_dish": None,
            "expect_category": category,
            "expect_difficulty": difficulty,
            "sample_dish": sample,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"写入 {len(rows)} 条评测问题 -> {OUT}")


if __name__ == "__main__":
    main()
