import os
import mlflow
from typing import List

from pydantic import BaseModel
from fastapi import FastAPI

codification_ape_app = FastAPI()


class Liasse(BaseModel):
    text_description: str
    type_: str
    nature: str
    surface: str
    event: str


@codification_ape_app.get("/")
def show_welcome_page():
    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")
    return {
        "Message": "Codification de l'APE",
        "Model_name": f"{model_name}",
        "Model_version": f"{model_version}",
    }


@codification_ape_app.get("/liasse")
def get_code_APE(
    text_feature: str,
    type_liasse: str | None = None,
    nature: str | None = None,
    surface: str | None = None,
    event: str | None = None,
    k: int = 2,
):
    query = {
        "query": {
            "TEXT_FEATURE": [text_feature],
            "AUTO": [type_liasse],
            "NAT_SICORE": [nature],
            "SURF": [surface],
            "EVT_SICORE": [event],
        },
        "k": k,
    }

    model = get_model()
    res = model.predict(query)
    return res[0][0][0].replace("__label__", "")


@codification_ape_app.post("/liasse")
def post_code_APE(liasse: Liasse):
    query = {
        "query": {
            "TEXT_FEATURE": [liasse.text_description],
            "AUTO": [liasse.type_],
            "NAT_SICORE": [liasse.nature],
            "SURF": [liasse.surface],
            "EVT_SICORE": [liasse.event],
        },
        "k": k,
    }

    model = get_model()
    res = model.predict(query)
    return res[0][0][0].replace("__label__", "")


@codification_ape_app.post("/liasses")
def get_list_code_APE(liasses: List[Liasse], k: int):
    query = {
        "query": {
            "TEXT_FEATURE": [liasse.text_description for liasse in liasses],
            "AUTO": [liasse.type_ for liasse in liasses],
            "NAT_SICORE": [liasse.nature for liasse in liasses],
            "SURF": [liasse.surface for liasse in liasses],
            "EVT_SICORE": [liasse.event for liasse in liasses],
        },
        "k": k,
    }

    model = get_model()
    res = model.predict(query)
    return res[0][0][0].replace("__label__", "")


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
            f"Failed to fetch model {model_name} version {model_version}: {str(error)}"
        ) from error
