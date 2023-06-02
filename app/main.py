"""
Main file for the API.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mlflow import MlflowClient
from pydantic import BaseModel

from app.utils import (
    get_model,
    preprocess_batch,
    preprocess_query,
    process_response,
)


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


class Liasses(BaseModel):
    """
    Pydantic BaseModel for representing the input data for the API.

    This BaseModel defines the structure of the input data required
    for the API's "/predict-batch" endpoint.

    Attributes:
        text_description (List[str]): The text description.
        type_ (List[str]): The type of liasse.
        nature (List[str]): The nature of the liasse.
        surface (List[str]): The surface of the liasse.
        event (List[str]): The event of the liasse.

    """

    text_description: List[str]
    type_: List[str]
    nature: List[str]
    surface: List[str]
    event: List[str]

    class Config:
        schema_extra = {
            "example": {
                "text_description": [
                    (
                        "LOUEUR MEUBLE NON PROFESSIONNEL EN RESIDENCE DE "
                        "SERVICES (CODE APE 6820A Location de logements)"
                    )
                ],
                "type_": ["I"],
                "nature": [""],
                "surface": [""],
                "event": ["01P"],
            }
        }


class LiassesEvaluation(BaseModel):
    """
    Pydantic BaseModel for representing the input data for the API.

    This BaseModel defines the structure of the input data required
    for the API's "/evaluation" endpoint.

    Attributes:
        text_description (List[str]): The text description.
        type_ (List[str]): The type of liasse.
        nature (List[str]): The nature of the liasse.
        surface (List[str]): The surface of the liasse.
        event (List[str]): The event of the liasse.
        code (List[str]): The true code of the liasse.

    """

    text_description: List[str]
    type_: List[str]
    nature: List[str]
    surface: List[str]
    event: List[str]
    code: List[str]

    class Config:
        schema_extra = {
            "example": {
                "text_description": [
                    (
                        "LOUEUR MEUBLE NON PROFESSIONNEL EN RESIDENCE DE "
                        "SERVICES (CODE APE 6820A Location de logements)"
                    )
                ],
                "type_": ["I"],
                "nature": [""],
                "surface": [""],
                "event": ["01P"],
                "code": ["6820A"],
            }
        }


codification_ape_app = FastAPI(
    lifespan=lifespan,
    title="Prédiction code APE",
    description="Application de prédiction pour \
                                            l'activité principale \
                                            de l'entreprise (APE)",
    version="0.0.1",
)


codification_ape_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@codification_ape_app.get("/", tags=["Welcome"])
def show_welcome_page():
    """
    Show welcome page with model name and version.
    """
    client = MlflowClient()
    run = client.get_run(model.metadata.run_id)
    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")
    metrics = {
        key: "Passed"
        if "Result" in key and value == 1
        else "Failed"
        if "Result" in key and value == 0
        else value
        for key, value in run.data.metrics.items()
    }

    return {
        "Message": "Codification de l'APE",
        "Model_name": f"{model_name}",
        "Model_version": f"{model_version}",
    } | {"Metrics": metrics}


@codification_ape_app.get("/predict", tags=["Predict"])
async def predict(
    text_feature: str,
    type_liasse: str | None = None,
    nature: str | None = None,
    surface: str | None = None,
    event: str | None = None,
    nb_echos_max: int = 5,
    prob_min: float = 0.01,
):
    """
    Predict code APE.

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

    response = process_response(predictions, 0, nb_echos_max, prob_min, libs)

    return response


@codification_ape_app.post("/predict-batch", tags=["Predict"])
async def predict_batch(
    liasses: Liasses,
    nb_echos_max: int = 5,
    prob_min: float = 0.01,
):
    """


    Args:


    Returns:
        dict: Response containing APE codes.
    """

    #
    query = preprocess_batch(liasses.dict(), nb_echos_max)

    predictions = model.predict(query)

    response = [
        process_response(predictions, i, nb_echos_max, prob_min, libs)
        for i in range(len(predictions[0]))
    ]

    return response


@codification_ape_app.post("/evaluation", tags=["Evaluate"])
async def eval_batch(
    liasses: LiassesEvaluation,
):
    """


    Args:


    Returns:
        dict: Response containing APE codes.
    """

    query = preprocess_batch(liasses.dict(), nb_echos_max=2)

    predictions = model.predict(query)

    df = pd.DataFrame(
        [
            [
                predictions[1][i][0],
                np.diff(predictions[1][i])[0] * -1,
                predictions[0][i][0].replace("__label__", ""),
            ]
            for i in range(len(predictions[0]))
        ],
        columns=["IC", "Probability", "Prediction"],
    )

    df["Code"] = liasses.code
    df["Result"] = df["Code"] == df["Prediction"]

    return df.to_dict()


# LOG libellé pré traité
