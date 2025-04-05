import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

# "The similarity score between S1 and S2 should be 4.5.\n - The similarity score between S1 and S3 should be 2.5."


ds = load_dataset("kardosdrur/synthetic-nordic-unit_triple")

splits = {}
for language in ds:
    s1 = []
    s2 = []
    score = []
    instruction = []
    for record in ds[language]:
        # Adding one record for the first pair
        s1.append(record["S1"])
        s2.append(record["S2"])
        score.append(4.5)
        instruction.append(record["instruction"])
        # Adding one record for the second pair
        s1.append(record["S1"])
        s2.append(record["S3"])
        score.append(2.5)
        instruction.append(record["instruction"])
    # Converting to a score between 0 and 1
    score = np.array(score) / 5
    splits[language] = Dataset.from_dict(
        {"s1": s1, "s2": s2, "score": score, "instruction": instruction}
    )
new_ds = DatasetDict(splits)

new_ds.push_to_hub("kardosdrur/synthetic-nordic-sts")
