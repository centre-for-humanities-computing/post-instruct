from pathlib import Path

import numpy as np
from autokeras import AutoModel, Input, RegressionHead
from datasets import load_dataset
from joblib import dump
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score

if not Path("dat/embeddings.npz").exists():
    ds = load_dataset("kardosdrur/post-instruct-queries")
    X_train = np.concatenate(
        (ds["train"]["instruction_embedding"], ds["train"]["query_embedding"]), axis=1
    )
    y_train = np.array(ds["train"]["joint_embedding"])
    X_validation = np.concatenate(
        (
            ds["validation"]["instruction_embedding"],
            ds["validation"]["query_embedding"],
        ),
        axis=1,
    )
    y_validation = np.array(ds["validation"]["joint_embedding"])
    embeddings = {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
    }
    np.savez("dat/embeddings.npz", **embeddings)
else:
    embeddings = np.load("dat/embeddings.npz")
    X_train = embeddings["X_train"]
    y_train = embeddings["y_train"]
    X_validation = embeddings["X_validation"]
    y_validation = embeddings["y_validation"]

regression_model = AutoModel(
    inputs=Input(),
    outputs=RegressionHead(),
)
regression_model.fit(X_train, y_train)

Path("models").mkdir(exist_ok=True)
dump(regression_model, "models/auto_model.joblib")

dummy_model = DummyRegressor()
dummy_model.fit(X_train, y_train)

print("Reg: ", r2_score(y_validation, regression_model.predict(X_validation)))
print("Dummy: ", r2_score(y_validation, dummy_model.predict(X_validation)))
