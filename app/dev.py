import os
from pathlib import Path

import mlflow
import yaml


def get_model():
    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")

    try:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )
        return model
    except Exception as error:
        raise Exception(
            f"Failed to fetch model {model_name} version \
                 {model_version}: {str(error)}"
        ) from error


# %env MLFLOW_MODEL_NAME=test
# %env MLFLOW_MODEL_VERSION=2
# %env AWS_ACCESS_KEY_ID=
# %env AWS_SECRET_ACCESS_KEY=
# %env MLFLOW_S3_ENDPOINT_URL=
# %env MLFLOW_TRACKING_URI=https://projet-ape-654210.user.lab.sspcloud.fr


del os.environ["AWS_SESSION_TOKEN"]

model = get_model()

query = {
    "query": {
        "TEXT_FEATURE": ["text_feature"],
        "AUTO": ["type_liasse"],
        "NAT_SICORE": ["nature"],
        "SURF": ["surface"],
        "EVT_SICORE": ["event"],
    },
    "k": 5,
}

res = model.predict(query)
libs = {}
libs["lib"] = yaml.safe_load(Path("libs.yaml").read_text())

output_dict = {
    rank_pred
    + 1: {
        "code": res[0][0][rank_pred].replace("__label__", ""),
        "probabilite": res[1][0][rank_pred],
        "libelle": libs["lib"][res[0][0][rank_pred].replace("__label__", "")],
    }
    for rank_pred in range(query["k"])
}
