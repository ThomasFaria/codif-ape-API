"""
Main file for the API.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from pydantic import BaseModel

from app.utils import get_model, preprocess_query, process_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for managing the lifespan of the API.

    This context manager is used to load the ML model and other resources
    when the API starts and clean them up when the API stops.

    Args:
        app (FastAPI): The FastAPI application.
    """
    global model, libs

    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")
    # Load the ML model
    model = get_model(model_name, model_version)
    libs = yaml.safe_load(Path("app/libs.yaml").read_text())

    yield
    # Clean up the ML models and release the resources
    model.clear()


class Liasse(BaseModel):
    """
    Pydantic BaseModel for representing the input data for the API.

    This BaseModel defines the structure of the input data required
    for the API's "/liasse" endpoint.

    Attributes:
        text_description (str): The text description.
        type_ (str): The type of liasse.
        nature (str): The nature of the liasse.
        surface (str): The surface of the liasse.
        event (str): The event of the liasse.

    """

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

    This endpoint accepts input data as query parameters and uses the loaded
    ML model to predict the code APE based on the input data.

    Args:
        text_feature (str): The text feature.
        type_liasse (str, optional): The type of liasse. Defaults to None.
        nature (str, optional): The nature of the liasse. Defaults to None.
        surface (str, optional): The surface of the liasse. Defaults to None.
        event: (str, optional): Event of the liasse. Optional.
        nb_echos_max (int): Maximum number of echoes to consider. Default is 5.
        prob_min (float): Minimum probability threshold. Default is 0.01.

    Returns:
        dict: Response containing APE codes.
    """

    query = preprocess_query(
        text_feature, type_liasse, nature, surface, event, nb_echos_max
    )

    predictions = model.predict(query)

    response = process_response(predictions, nb_echos_max, prob_min, libs)

    return response


# TODO: mettre des fonction globales
