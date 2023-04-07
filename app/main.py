import os
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI
from pydantic import BaseModel

ml_models = {}


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["fastText"] = get_model
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


class Liasse(BaseModel):
    text_description: str
    type_: str
    nature: str
    surface: str
    event: str


codification_ape_app = FastAPI(lifespan=lifespan)


@codification_ape_app.get("/", tags=["Welcome"])
def show_welcome_page():
    """
    Show welcome page with model name and version.
    """
    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")
    return {
        "Message": "Codification de l'APE",
        "Model_name": f"{model_name}",
        "Model_version": f"{model_version}",
    }


@codification_ape_app.get("/liasse", tags=["Liasse"])
async def get_code_APE(
    text_feature: str,
    type_liasse: str | None = None,
    nature: str | None = None,
    surface: str | None = None,
    event: str | None = None,
    k: int = 2,
):
    """
    Get code APE from ML model.
    """
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

    res = ml_models["fastText"].predict(query)
    return res[0][0][0].replace("__label__", "")
