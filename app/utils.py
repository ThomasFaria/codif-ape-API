import mlflow
import numpy as np
from fastapi.responses import JSONResponse


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


def preprocess_query(
    text_feature: str,
    type_liasse: str | None,
    nature: str | None,
    surface: str | None,
    event: str | None,
    nb_echos_max: int = 5,
):
    type_liasse, nature, surface, event = (
        "NaN" if v is None else v
        for v in (type_liasse, nature, surface, event)
    )

    query = {
        "query": {
            "TEXT_FEATURE": [text_feature],
            "AUTO": [type_liasse],
            "NAT_SICORE": [nature],
            "SURF": [surface],
            "EVT_SICORE": [event],
        },
        "k": nb_echos_max,
    }
    return query


def process_response(
    predictions: tuple,
    nb_echos_max: int,
    prob_min: float,
    libs: dict,
):
    k = nb_echos_max
    if predictions[1][0][-1] < prob_min:
        k = np.min(
            [np.argmax(not (predictions[1][0] > prob_min)), nb_echos_max]
        )

    output_dict = {
        rank_pred
        + 1: {
            "code": predictions[0][0][rank_pred].replace("__label__", ""),
            "probabilite": float(predictions[1][0][rank_pred]),
            "libelle": libs["lib"][
                predictions[0][0][rank_pred].replace("__label__", "")
            ],
        }
        for rank_pred in range(k)
    }

    try:
        response = output_dict | {
            "IC": output_dict[1]["probabilite"] - float(predictions[1][0][1])
        }
        return response
    except KeyError:
        return JSONResponse(
            status_code=404,
            content={
                "message": "The minimal probability requested is higher "
                "than the highest prediction probability of the model."
            },
        )
