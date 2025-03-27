from pathlib import Path

import mteb
import numpy as np
from mteb.encoder_interface import Encoder, PromptType

from post_instruct.model import PostInstruct


class PostInstructWrapper(Encoder):
    def __init__(self, model_path: str | Path):
        super().__init__()
        self.model = PostInstruct(model_path)

    def encode(
        self,
        sentences: list[str],
        *,
        prompt_type: PromptType | None = None,
        task_name: str,
        **kwargs,
    ) -> np.ndarray:
        prompt = mteb.get_task(task_name).metadata.prompt
        embeddings = self.model.encode(sentences, instruction=prompt)
        return embeddings


task = mteb.get_task("ArXivHierarchicalClusteringS2S")
evaluation = mteb.MTEB(tasks=[task])
evaluation.run(
    PostInstructWrapper("models/all-MiniLM-L6-v2-post-instruct/checkpoint-675"),
    output_folder="results",
)
