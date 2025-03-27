import json
import os
from typing import Optional

import numpy as np
import torch
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model
from sentence_transformers import (SentenceTransformer,
                                   SentenceTransformerTrainer)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch import Tensor, nn


class InstructAdaptor(nn.Sequential):
    def __init__(self, embedding_size: int):
        self.embedding_size = embedding_size
        super().__init__(
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.LeakyReLU(),
            nn.Linear(self.embedding_size, self.embedding_size**2),
            nn.Unflatten(1, torch.Size([self.embedding_size, self.embedding_size])),
        )

    def get_config_dict(self) -> dict[str, float]:
        return {"embedding_size": self.embedding_size}

    def get_sentence_embedding_dimension(self) -> int:
        return self.embedding_size

    def save(self, output_path, safe_serialization: bool = True) -> None:
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut)
        if safe_serialization:
            save_safetensors_model(self, os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(
                self.state_dict(), os.path.join(output_path, "pytorch_model.bin")
            )

    @staticmethod
    def load(input_path) -> "InstructAdaptor":
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)
        model = InstructAdaptor(**config)
        if os.path.exists(os.path.join(input_path, "model.safetensors")):
            load_safetensors_model(model, os.path.join(input_path, "model.safetensors"))
        else:
            model.load_state_dict(
                torch.load(
                    os.path.join(input_path, "pytorch_model.bin"),
                    map_location=torch.device("cpu"),
                    weights_only=True,
                )
            )
        return model


class PostInstruct(SentenceTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_size = self._first_module().get_word_embedding_dimension()
        self.adaptor = InstructAdaptor(self.embedding_size)
        self.adaptor.to(self.device)
        for module_name, module in self.named_children():
            if module_name == "adaptor":
                continue
            else:
                module.requires_grad = False

    def encode(
        self,
        sentences: str | list[str] | np.ndarray,
        prompt_name: str | None = None,
        prompt: str | None = None,
        *args,
        instruction: str | None = None,
        **kwargs
    ):
        instruction = instruction or prompt
        if (
            (instruction is None)
            and (prompt_name is not None)
            and (self.prompts is not None)
        ):
            instruction = self.prompts.get(prompt_name, None)
        return super().encode(sentences, *args, instruction=instruction, **kwargs)

    def apply_instruction(
        self, embeddings: Tensor | np.ndarray, instruction: str
    ) -> np.ndarray:
        with torch.no_grad():
            inst_emb = self.encode([instruction], convert_to_tensor=True)
            inst_mat = self.adaptor.forward(inst_emb)
            embeddings = np.array(embeddings) @ np.array(inst_mat)
        return embeddings

    def forward(
        self, features: dict[str, Tensor], instruction: Optional[str] = None, **kwargs
    ):
        inst_mat = None
        if "instruction_embedding" in features:
            inst_mat = self.adaptor.forward(features["instruction_embedding"])[0]
        if instruction is not None:
            #  Encode instruction with no grad
            inst_emb = super().encode([instruction], convert_to_tensor=True)
            inst_mat = self.adaptor.forward(inst_emb)[0]
        if self.module_kwargs is None:
            for module_name, module in self.named_children():
                if module_name == "adaptor":
                    continue
                features = module(features)
        else:
            for module_name, module in self.named_children():
                if module_name == "adaptor":
                    continue
                module_kwarg_keys = self.module_kwargs.get(module_name, [])
                module_kwargs = {
                    key: value
                    for key, value in kwargs.items()
                    if key in module_kwarg_keys
                }
                features = module(features, **module_kwargs)
        if inst_mat is None:
            return features
        print("Applying instructions")
        embeddings = features["sentence_embedding"]
        embeddings = embeddings @ inst_mat
        features["sentence_embedding"] = embeddings
        return features
