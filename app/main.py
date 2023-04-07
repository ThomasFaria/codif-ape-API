import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from utils import get_model, preprocess_query, process_response

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
    nb_echos_max: int = 5,
    prob_min: float = 0.01,
):
    """
    Get code APE.
    """

    query = preprocess_query(
        text_feature, type_liasse, nature, surface, event, nb_echos_max
    )

    predictions = ml_models["fastText"].predict(query)

    response = process_response(predictions, nb_echos_max, prob_min, libs)

    return response


# TODO: Creer un utils.py qui contient toutes les fonction python,
# ici garder les fonctions le
# simple possible. Et garder que les fonctions de l'API
# TODO: mettre des fonction globales
