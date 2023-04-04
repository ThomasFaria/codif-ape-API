import os

from fastapi import FastAPI

codification_ape_app = FastAPI()


@codification_ape_app.get("/")
def show_welcome_page():
    model_name: str = os.getenv("MLFLOW_MODEL_NAME")
    model_version: str = os.getenv("MLFLOW_MODEL_VERSION")
    return {
        "Message": "Codification de l'APE",
        "Model_name": f"{model_name}",
        "Model_version": f"{model_version}",
    }

