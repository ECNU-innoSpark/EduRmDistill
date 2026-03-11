from datasets import load_dataset
import json
from pathlib import Path

HERE = Path(__file__).parent

# 以 Alpaca 为例，只取前 500 条
ds = load_dataset("yahma/alpaca-cleaned", split="train[:10]")

with open(HERE.parent / "data" / "raw_prompts.jsonl", "w", encoding="utf-8") as f:
    for item in ds:
        # 构造符合你脚手架格式的 JSON
        line = {"messages": [{"role": "user", "content": item["instruction"]}]}  # pyright: ignore[reportCallIssue, reportArgumentType]
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

print("种子数据已准备就绪！")
