import ast
import json
from collections import defaultdict

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import get_collection

TASK_TYPE_TO_VALID_COLUMNS = {
    "retrieval": {
        "user_query",
        "hard_negative_document",
        "positive_document",
        "instruction",
    },
    "classification": {"input_text", "label", "misleading_label", "instruction"},
    "unit_triple": {"S1", "S2", "S3", "instruction"},
    "text_matching": {"input", "positive_document", "instruction"},
}


def parse_response(response: str):
    response = response.split("{", 1)[-1].rsplit("}", 1)[0]
    response = "{" + response + "}"
    return json.loads(" ".join(response.split()))


def parse_instruction(prompt: list[dict[str, str]]) -> str:
    content = prompt[0]["content"]
    instruction = content.split("task:", 1)[-1].split("\n", 1)[0]
    return instruction


def parse_record(example: dict) -> dict:
    return {
        **parse_response(example["response"]),
        "instruction": parse_instruction(example["prompt"]),
    }


def process_dataset(ds: Dataset, task_type: str) -> Dataset:
    def _generate_dataset():
        for record in ds:
            try:
                record = parse_record(record)
                if set(record.keys()) == TASK_TYPE_TO_VALID_COLUMNS[task_type]:
                    yield record
                else:
                    columns = ", ".join(record.keys())
                    raise ValueError(f"Columns are invalid: {columns}")
            except Exception as e:
                print(f"[WARNING] Couldn't parse record due to exception: {e}")

    records = list(_generate_dataset())
    df = pd.DataFrame.from_records(records)
    return Dataset.from_pandas(df)


print("Loading collection")
collection = get_collection(
    "ThatsGroes/nordic-embedding-training-data-678f53542163a7eaf5d2194e"
)

tasks_per_type = defaultdict(list)
for elem in collection.items:
    task_type = "_".join(
        elem.item_id.removeprefix("ThatsGroes/synthetic-from-").split("-")[:2]
    ).removesuffix("_tasks")
    if task_type == "text_mathing":
        task_type = "text_matching"
    tasks_per_type[task_type].append(elem.item_id)

for task_type, dataset_ids in tasks_per_type.items():
    print("------------------------")
    print(f"Processing {task_type} tasks")
    print("------------------------")
    splits = {}
    by_language = defaultdict(list)
    for dataset_id in dataset_ids:
        language = dataset_id.split("-")[-1]
        by_language[language].append(dataset_id)
    for language, ds_ids in by_language.items():
        print(f" - Processing language: {language}...")
        if len(ds_ids) == 1:
            ds = load_dataset(ds_ids[0])["train"]
        else:
            ds = concatenate_datasets(
                [load_dataset(ds_id)["train"] for ds_id in ds_ids]
            )
        ds = process_dataset(ds, task_type=task_type)
        splits[language] = ds
    task_dataset = DatasetDict(splits)
    repo_id = f"kardosdrur/synthetic-nordic-{task_type}"
    print(f"Saving {task_type} tasks to {repo_id}")
    task_dataset.push_to_hub(repo_id)
print("DONE")
