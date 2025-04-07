import mteb
from collections import Counter
from sentence_transformers.evaluation import NanoBEIREvaluator
from sentence_transformers.trainer import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers

from post_instruct.datasets import prepare_mteb_tasks
from post_instruct.model import PostInstruct
from post_instruct.training import PostInstructTrainer, load_default_instructions

print("Initializing model")
model = PostInstruct("all-MiniLM-L6-v2")

print("Selecting English tasks")
task_names = list(load_default_instructions().keys())
english_tasks = mteb.get_tasks(languages=["eng"], exclusive_language_filter=True)
english_task_names = [task.metadata.name for task in english_tasks]
overlap = list(set(task_names) & set(english_task_names))
tasks = mteb.get_tasks(tasks=overlap, languages=["eng"], exclusive_language_filter=True)

print("Loading tasks for training.")
training_datasets, losses = prepare_mteb_tasks(tasks, model)

print(f"{len(training_datasets)} datasets collected in total:")
training_tasks = mteb.get_tasks(tasks=list(training_datasets.keys()))
task_types = Counter([task.metadata.type for task in training_tasks])
for task_type, n in task_types.items():
    print(f"  - {task_type}: {n}")


print("Initializing evaluator")
# Only using MSMARCO, since all-MiniLM-L6-v2 has already been trained on it,
# but I don't want to see performance on other MTEB(eng) tasks while training
evaluator = NanoBEIREvaluator(
    dataset_names=["MSMARCO"],
    query_prompts={
        "MSMARCO": "Given a web search query, retrieve relevant passages that answer the query"
    },
)
print("Evaluating base model.")
evaluator(model)

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/all-MiniLM-L6-v2-post-instruct",
    # Optional training parameters:
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
    # Optional tracking/debugging parameters:
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="all-MiniLM-L6-v2-post-instruct",  # Will be used in W&B if `wandb` is installed
)

print("Initializing Trainer")
trainer = PostInstructTrainer(
    model=model,
    args=args,
    train_dataset=training_datasets,
    loss=losses,
    evaluator=evaluator,
)

print("Training")
trainer.train()

print("Done")
