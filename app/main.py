import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

ml_models = {}
libs = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")
    # Load the ML model
    ml_models["fastText"] = get_model(model_name, model_version)
    libs["lib"] = yaml.safe_load(Path("app/libs.yaml").read_text())

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
    Get code APE.
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

    output_dict = {
        rank_pred
        + 1: {
            "code": res[0][0][rank_pred].replace("__label__", ""),
            "probabilite": float(res[1][0][rank_pred]),
            "libelle": libs["lib"][
                res[0][0][rank_pred].replace("__label__", "")
            ],
        }
        for rank_pred in range(query["k"])
    }
    print(output_dict)
    return output_dict


def get_model(model_name: str, model_version: str):
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
