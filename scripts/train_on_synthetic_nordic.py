import os

from datasets import load_dataset
from sentence_transformers.evaluation import NanoBEIREvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.trainer import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers

from post_instruct.collator import PostInstructDataCollator
from post_instruct.model import PostInstruct
from post_instruct.training import PostInstructTrainer

base_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model_name = "nordic-MiniLM-L12-post-instruct"

os.environ["WANDB_PROJECT"] = model_name  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

print("Initializing model")
model = PostInstruct(base_model)

print("Loading tasks for training.")
training_datasets = {}
losses = {}

retrieval_ds = load_dataset("kardosdrur/synthetic-nordic-retrieval")
for language in retrieval_ds:
    task_name = f"retrieval_{language}"
    training_datasets[task_name] = retrieval_ds[language]
    losses[task_name] = MultipleNegativesRankingLoss(model)
classification_ds = load_dataset(
    "kardosdrur/synthetic-nordic-classification"
).rename_column("label", "correct_label")
for language in classification_ds:
    task_name = f"classification_{language}"
    training_datasets[task_name] = classification_ds[language]
    losses[task_name] = MultipleNegativesRankingLoss(model)
bitext_ds = load_dataset("kardosdrur/synthetic-nordic-text_matching")
for language in bitext_ds:
    task_name = f"bitext_{language}"
    training_datasets[task_name] = bitext_ds[language]
    losses[task_name] = MultipleNegativesRankingLoss(model)

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
print(evaluator(model))

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{model_name}",
    # Optional training parameters:
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
    eval_strategy="steps",
    eval_steps=100,
    # Optional tracking/debugging parameters:
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name=model_name,  # Will be used in W&B if `wandb` is installed
    report_to="wandb",
)

print("Initializing Trainer")
trainer = PostInstructTrainer(
    model=model,
    args=args,
    train_dataset=training_datasets,
    loss=losses,
    evaluator=evaluator,
    data_collator=PostInstructDataCollator(tokenize_fn=model.tokenize),
)

print("Training")
trainer.train()

print("Saving")
model.save_pretrained("models/model_name/final")

print("Done")
