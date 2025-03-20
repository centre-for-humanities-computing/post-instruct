import itertools
import json
import random
from functools import partial
from pathlib import Path
from typing import Iterable

from datasets import Dataset, DatasetDict
from mteb.abstasks.AbsTask import AbsTask
from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from mteb.abstasks.AbsTaskSummarization import AbsTaskSummarization
from tqdm import tqdm

from utils import load_tasks


def stream_queries(task: AbsTask) -> Iterable[dict]:
    task.load_data()
    hf_subsets = list(task.dataset) if task.is_multilingual else ["default"]
    dataset = task.dataset
    if isinstance(task, (AbsTaskBitextMining, AbsTaskSTS, AbsTaskPairClassification)):
        for hf_subset in hf_subsets:
            if hf_subset != "default":
                ds = dataset[hf_subset]
            else:
                ds = dataset
            splits = ds.keys()
            for split in splits:
                sentences = ds[split]["sentence1"] + dataset[split]["sentence2"]
                for sentence in sentences:
                    yield dict(
                        task_name=task.metadata.name,
                        subset=hf_subset,
                        split=split,
                        query=sentence,
                    )
    elif isinstance(task, (AbsTaskClassification, AbsTaskSummarization)):
        for hf_subset in hf_subsets:
            if hf_subset != "default":
                ds = dataset[hf_subset]
            else:
                ds = dataset
            splits = ds.keys()
            for split in splits:
                sentences = ds[split]["text"]
                for sentence in sentences:
                    yield dict(
                        task_name=task.metadata.name,
                        subset=hf_subset,
                        split=split,
                        query=sentence,
                    )
    elif isinstance(task, AbsTaskClusteringFast):
        for hf_subset in hf_subsets:
            if hf_subset != "default":
                ds = dataset[hf_subset]
            else:
                ds = dataset
            splits = ds.keys()
            for split in splits:
                sentences = ds[split]["sentences"]
                for sentence in sentences:
                    yield dict(
                        task_name=task.metadata.name,
                        subset=hf_subset,
                        split=split,
                        query=sentence,
                    )
    elif isinstance(task, AbsTaskClustering):
        for hf_subset in hf_subsets:
            if hf_subset != "default":
                ds = dataset[hf_subset]
            else:
                ds = dataset
            splits = ds.keys()
            for split in splits:
                sentences = itertools.chain.from_iterable(ds[split]["sentences"])
                for sentence in sentences:
                    yield dict(
                        task_name=task.metadata.name,
                        subset=hf_subset,
                        split=split,
                        query=sentence,
                    )
    elif isinstance(task, AbsTaskReranking):
        for hf_subset in hf_subsets:
            if hf_subset != "default":
                ds = dataset[hf_subset]
            else:
                ds = dataset
            splits = ds.keys()
            for split in splits:
                sentences = ds[split]["query"]
                for sentence in sentences:
                    yield dict(
                        task_name=task.metadata.name,
                        subset=hf_subset,
                        split=split,
                        query=sentence,
                    )
    elif isinstance(task, AbsTaskRetrieval):
        hf_subsets = list(task.hf_subsets) if task.is_multilingual else ["default"]
        for hf_subset in hf_subsets:
            if hf_subset != "default":
                qs = task.queries[hf_subset]
            else:
                qs = task.queries
            splits = qs.keys()
            for split in splits:
                for q_id, sentences in qs[split].items():
                    if isinstance(sentences, str):
                        sentences = [sentences]
                    for sentence in sentences:
                        yield dict(
                            task_name=task.metadata.name,
                            subset=hf_subset,
                            split=split,
                            query=sentence,
                        )
    else:
        raise TypeError("Task not in supported task categories.")


def generate_all_entries(instructions: dict[str, list[str]]) -> Iterable[dict]:
    random.seed(42)
    tasks = load_tasks()
    for task in tqdm(tasks, desc="Processing all tasks."):
        if task.metadata.name not in instructions:
            continue
        try:
            entries = itertools.islice(stream_queries(task), 1000)
            possible_instructions = instructions[task.metadata.name]
            for entry in entries:
                # Adding a random instruction to the entry
                yield {**entry, "instruction": random.choice(possible_instructions)}
        except Exception as e:
            print(
                f"[WARNING] Couldn't collect entries for task {task.metadata.name} due to error: {e}"
            )
            continue


def main():
    with Path("dat/synthetic_instructions.json").open() as inst_file:
        instructions = json.loads(inst_file.read())
    with Path("dat/dataset.jsonl").open("a") as out_file:
        for entry in generate_all_entries(instructions):
            out_file.write(json.dumps(entry) + "\n")
    print("Done")


if __name__ == "__main__":
    main()
