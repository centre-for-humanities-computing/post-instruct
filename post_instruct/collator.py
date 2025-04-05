from typing import Any

import torch
from sentence_transformers.data_collator import SentenceTransformerDataCollator


class PostInstructDataCollator(SentenceTransformerDataCollator):

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        column_names = list(features[0].keys())

        # We should always be able to return a loss, label or not:
        batch = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        if "instruction" in column_names:
            column_names.remove("instruction")
            batch["instruction"] = [record["instruction"] for record in features]

        if tuple(column_names) not in self._warned_columns:
            self.maybe_warn_about_column_order(column_names)

        # Extract the label column if it exists
        for label_column in self.valid_label_columns:
            if label_column in column_names:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        for column_name in column_names:
            # If the prompt length has been set, we should add it to the batch
            if (
                column_name.endswith("_prompt_length")
                and column_name[: -len("_prompt_length")] in column_names
            ):
                batch[column_name] = torch.tensor(
                    [row[column_name] for row in features], dtype=torch.int
                )
                continue

            tokenized = self.tokenize_fn([row[column_name] for row in features])
            for key, value in tokenized.items():
                batch[f"{column_name}_{key}"] = value

        return batch
