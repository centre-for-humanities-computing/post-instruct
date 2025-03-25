import mteb
import numpy as np
from keras.saving import load_model
from mteb.encoder_interface import Encoder, PromptType
from sentence_transformers import SentenceTransformer


class PostInstruct(Encoder):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
        self.adaptor = load_model("auto_model/best_model.keras")

    def encode(
        self,
        sentences: list[str],
        *,
        prompt_type: PromptType | None = None,
        task_name: str,
        **kwargs,
    ) -> np.ndarray:
        prompt = mteb.get_task(task_name).metadata.prompt
        if prompt_type == "query":
            query_embeddings = self.model.encode(
                [f"Query: {sentence}" for sentence in sentences]
            )
            instruction_embedding = self.model.encode([f"Instruction: {prompt}"])
            instruction_embedding = np.broadcast_to(
                instruction_embedding,
                (query_embeddings.shape[0], instruction_embedding.shape[1]),
            )
            embeddings = self.adaptor.predict(
                np.concatenate((instruction_embedding, query_embeddings), axis=1)
            )
            embeddings = np.array(embeddings)
        else:
            embeddings = self.model.encode(sentences)
        return embeddings


task = mteb.get_task("SwednClusteringS2S")
evaluation = mteb.MTEB(tasks=[task])
evaluation.run(PostInstruct(), output_folder="results")
