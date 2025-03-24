import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict


def stream_dataset() -> Iterable[dict]:
    with Path("dat/dataset.jsonl").open() as in_file:
        for line in in_file:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    entries = stream_dataset()
    split_dict = defaultdict(list)
    for entry in entries:
        split = entry.pop("split")
        split_dict[split].append(entry)
    ds = DatasetDict(
        {
            split: Dataset.from_generator(lambda: iter(entries))
            for split, entries in split_dict.items()
        }
    )
    ds.shuffle(seed=42)
    ds.push_to_hub("kardosdrur/post-instruct-queries")
    print("DONE")


if __name__ == "__main__":
    main()
