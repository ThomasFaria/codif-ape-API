import os
import mlflow
import pandas as pd

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


@codification_ape_app.get("/liasse")
def get_code_APE(
    text_feature: str, type_liasse: str, nature: str, surface: str, event: str, k: int
):
    df = pd.DataFrame(
        columns=["text_feature", "type_liasse", "nature", "surface", "event"]
    )
    df.loc[0] = pd.Series(
        {
            "text_feature": text_feature,
            "type_liasse": type_liasse,
            "nature": nature,
            "surface": surface,
            "event": event,
        }
    )
    model = get_model()
    res = model.predict(df)
    return res


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
