from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def main():
    encoder = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

    def encode_batch(batch):
        joint_query_inst = []
        for instruction, query in zip(batch["instruction"], batch["query"]):
            joint_query_inst.append(f"Instruct: {instruction}\nQuery: {query}")
        inst_emb = encoder.encode(
            [f"Instruct: {instruction}" for instruction in batch["instruction"]]
        )
        query_emb = encoder.encode([f"Query: {query}" for query in batch["query"]])
        joint_emb = encoder.encode(joint_query_inst)
        return {
            "instruction_embedding": inst_emb.tolist(),
            "query_embedding": query_emb.tolist(),
            "joint_embedding": joint_emb.tolist(),
        }

    ds = load_dataset("kardosdrur/post-instruct-queries")
    ds.map(encode_batch, batched=True)
    ds.push_to_hub("kardosdrur/post-instruct-queries")


if __name__ == "__main__":
    main()
