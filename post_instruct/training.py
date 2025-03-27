import random
from typing import Any

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.trainer import SentenceTransformerTrainer


def load_default_instructions() -> dict[str, list[str]]:
    inst_ds = load_dataset("kardosdrur/synthetic_instructions")["train"]
    instructions = dict(zip(inst_ds["task_name"], inst_ds["instructions"]))
    return instructions


class PostInstructTrainer(SentenceTransformerTrainer):
    def __init__(
        self, *args, instructions: dict[str, list[str]] | None = None, **kwargs
    ) -> None:
        """Custom trainer implementation for post-instruct models.
        Accepts all arguments that SentenceTransformerTrainer does.

        Parameters
        ----------
        instructions: dict[str, list[str]] or None, default None
            Mapping of dataset name to a list of instructions to sample from.
            Defaults to the `kardosdrur/synthetic_instructions` dataset.
        """
        super().__init__(*args, **kwargs)
        self.instructions = (
            instructions if instructions is not None else load_default_instructions()
        )

    def compute_loss(
        self,
        model: SentenceTransformer,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        dataset_name = inputs.pop("dataset_name", None)
        if dataset_name in self.instructions:
            instruction = random.choice(self.instructions[dataset_name])
            inputs["instruction_embedding"] = model.encode(
                [instruction], convert_to_tensor=True
            )
        features, labels = self.collect_features(inputs)
        loss_fn = self.loss

        if isinstance(loss_fn, dict) and dataset_name:
            loss_fn = loss_fn[dataset_name]
        # Insert the wrapped (e.g. distributed or compiled) model into the loss function,
        # if the loss stores the model. Only called once per process
        if (
            model == self.model_wrapped
            and model != self.model  # Only if the model is wrapped
            and hasattr(loss_fn, "model")  # Only if the loss stores the model
            and loss_fn.model
            != model  # Only if the wrapped model is not already stored
        ):
            loss_fn = self.override_model_in_loss(loss_fn, model)
        loss = loss_fn(features, labels)
        if return_outputs:
            # During prediction/evaluation, `compute_loss` will be called with `return_outputs=True`.
            # However, Sentence Transformer losses do not return outputs, so we return an empty dictionary.
            # This does not result in any problems, as the SentenceTransformerTrainingArguments sets
            # `prediction_loss_only=True` which means that the output is not used.
            return loss, {}
        return loss
