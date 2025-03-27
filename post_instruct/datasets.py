import warnings

from datasets import Dataset, concatenate_datasets
from mteb.abstasks.AbsTask import AbsTask
from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from mteb.abstasks.AbsTaskClusteringFast import AbsTaskClusteringFast
from mteb.abstasks.AbsTaskPairClassification import AbsTaskPairClassification
from mteb.abstasks.AbsTaskReranking import AbsTaskReranking
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.AbsTaskSTS import AbsTaskSTS
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import (AnglELoss, BatchHardTripletLoss,
                                          MultipleNegativesRankingLoss,
                                          OnlineContrastiveLoss, SoftmaxLoss)
from torch import nn
from tqdm import tqdm

Loss = nn.Module
TaskName = str


def concat_multilingual(dataset, hf_subsets):
    if hf_subsets != ["default"]:
        datasets = [dataset[hf_subset]["train"] for hf_subset in hf_subsets]
        dataset = concatenate_datasets(datasets)
    else:
        dataset = dataset["train"]
    return dataset.shuffle(seed=42)


def unpack_negatives(example: dict) -> dict:
    res = {}
    res["anchor"] = example["query"]
    # Unpacking one positive example
    res["positive"], *_ = example["positive"]
    for i, negative in enumerate(example["negative"]):
        res[f"negative_{i}"] = negative
    return res


def prepare_retrieval(corpus, queries, relevant_docs, hf_subsets) -> Dataset:
    res_queries = []
    res_docs = []
    labels = []
    for hf_subset in hf_subsets:
        if hf_subset == "default":
            s_corpus, s_queries, s_docs = (
                corpus["train"],
                queries["train"],
                relevant_docs["train"],
            )
        else:
            s_corpus, s_queries, s_docs = (
                corpus[hf_subset]["train"],
                queries[hf_subset]["train"],
                relevant_docs[hf_subset]["train"],
            )
        for q_id, query in s_queries.items():
            if isinstance(query, list):
                raise TypeError(
                    "Reterieval tasks with multiple-turn conversations are not supported"
                )
            for doc_id, score in s_docs[q_id].items():
                res_queries.append(query)
                res_docs.append(s_corpus[doc_id]["text"])
                labels.append(score)
    dataset = Dataset.from_dict(
        {"query": res_queries, "document": res_docs, "label": labels}
    )
    dataset = dataset.shuffle(seed=42)
    return dataset


def prepare_mteb_task(
    task: AbsTask, model: SentenceTransformer
) -> tuple[TaskName, Dataset, Loss]:
    task.load_data(eval_splits=["train"])
    hf_subsets = list(task.dataset) if task.is_multilingual else ["default"]
    dataset = task.dataset
    if isinstance(task, AbsTaskSTS):
        loss = AnglELoss(model)
        dataset = concat_multilingual(dataset, hf_subsets)
        return task.metadata.name, dataset, loss
    elif isinstance(task, AbsTaskPairClassification):
        loss = SoftmaxLoss(model)
        dataset = concat_multilingual(dataset, hf_subsets)
        dataset = dataset.rename_column("labels", "label")
        return task.metadata.name, dataset, loss
    elif isinstance(task, AbsTaskBitextMining):
        loss = MultipleNegativesRankingLoss(model)
        dataset = concat_multilingual(dataset, hf_subsets)
        return task.metadata.name, dataset, loss
    elif isinstance(task, (AbsTaskClassification, AbsTaskClusteringFast)):
        loss = BatchHardTripletLoss(model)
        dataset = concat_multilingual(dataset, hf_subsets)
        if isinstance(task, AbsTaskClusteringFast):
            dataset = dataset.rename_column("labels", "label")
        if isinstance(dataset["label"][0], list):
            raise TypeError(
                f"{task.metadata.name}: Hierachical/Multilabel classification and clustering tasks are not supported."
            )
        return task.metadata.name, dataset, loss
    elif isinstance(task, AbsTaskClustering):
        raise TypeError(
            f"{task.metadata.name}: Old clustering tasks in MTEB are not supported."
        )
    elif isinstance(task, AbsTaskReranking):
        loss = MultipleNegativesRankingLoss(model)
        dataset = concat_multilingual(dataset)
        dataset = dataset.map(unpack_negatives)
        return task.metadata.name, dataset, loss
    elif isinstance(task, AbsTaskRetrieval):
        dataset = prepare_retrieval(
            task.corpus, task.queries, task.relevant_docs, hf_subsets
        )
        if set(dataset["label"]) == set([1]):
            # If there's only one label, the using a different loss make s a lot of sense
            loss = MultipleNegativesRankingLoss(model)
        else:
            loss = OnlineContrastiveLoss(model)
        return task.metadata.name, dataset, loss
    else:
        raise TypeError(f"{task.metadata.name}: Task not in supported task categories.")


def prepare_mteb_tasks(
    tasks: list[AbsTask],
    model: SentenceTransformer,
    max_examples: int | None = 10_000,
    show_progress_bar: bool = True,
) -> tuple[dict[TaskName, Dataset], dict[TaskName, Loss]]:
    """Converts MTEB tasks into datasets which are in the correct format for training,
    along with losses for each dataset.

    Parameters
    ----------
    tasks: list[AbsTask]
        List of MTEB tasks to convert into training format.
    model: SentenceTransformer
        Model to initialize losses with.
    max_examples: int or None, default 5000
        Maximum number of examples to load from each dataset.
    show_progress_bar: bool, default True
        Indicates whether a progress bar should be shown while loading the datasets.

    Returns
    -------
    training_datasets: dict[str, Dataset]
        Mapping from task names to datasets.
    losses: dict[str, nn.Module]
        Mapping from task names to losses.
    """
    datasets = dict()
    losses = dict()
    if show_progress_bar:
        tasks = tqdm(
            tasks, desc="Preparing all MTEB tasks to be used as training datasets."
        )
    for task in tasks:
        try:
            task_name, dataset, loss = prepare_mteb_task(task, model)
            first_column, *_ = dataset.features
            ds_size = len(dataset[first_column])
            if (max_examples is not None) and (ds_size > max_examples):
                # Cutting down the number of examples
                dataset = dataset.train_test_split(test_size=max_examples, seed=42)[
                    "test"
                ]
            datasets[task_name] = dataset
            losses[task_name] = loss
        except Exception as e:
            warnings.warn(
                f"Couldn't convert task, {task.metadata.name}, to training dataset due to exception: {e}"
            )
    return datasets, losses
