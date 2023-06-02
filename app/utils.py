import os
import re
from typing import Annotated

import mlflow
import numpy as np
import pandas as pd
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()


def get_model(model_name: str, model_version: str) -> object:
    """
    This function fetches a trained machine learning model from the MLflow
    model registry based on the specified model name and version.

    Args:
        model_name (str): The name of the model to fetch from the model
        registry.
        model_version (str): The version of the model to fetch from the model
        registry.

    Returns:
        model (object): The loaded machine learning model.

    Raises:
        Exception: If the model fetching fails, an exception is raised with an
        error message.
    """

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


def get_current_username(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)]
):
    print(credentials.username)
    print(credentials.password)
    print(os.getenv("API_USERNAME"))
    print(os.getenv("API_PASSWORD"))

    if not (credentials.username == os.getenv("API_USERNAME")) or not (
        credentials.password == os.getenv("API_PASSWORD")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentification failed",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def preprocess_query(
    text_feature: str,
    type_liasse: str | None,
    nature: str | None,
    surface: str | None,
    event: str | None,
    nb_echos_max: int = 5,
) -> dict:
    """
    This function preprocesses the input query parameters for making
    predictions using the fetched machine learning model.

    Args:
        text_feature (str): The text feature to be used for prediction.
        type_liasse (str | None): The type of liasse for the query.
        Can be None.
        nature (str | None): The nature of the liasse. Can be None.
        surface (str | None): The surface of the liasse. Can be None.
        event (str | None): The event of the liasse. Can be None.
        nb_echos_max (int, optional): The maximum number of echo predictions.
        Default is 5.

    Returns:
        query (dict): The preprocessed query in the required format for
        making predictions.

    """
    type_liasse, nature, surface, event = (
        np.nan if v is None else v
        for v in (type_liasse, nature, surface, event)
    )

    list_ok = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "G",
        "I",
        "L",
        "M",
        "N",
        "R",
        "S",
        "X",
        "Y",
        "Z",
    ]
    check_format_features(
        [type_liasse],
        "type_",
        r"^(" + "|".join(list_ok) + r")$",
        list_ok=list_ok,
    )

    check_format_features([nature], "nature", r"^\d{2}$")

    list_ok = ["1", "2", "3", "4"]
    check_format_features(
        [surface],
        "surface",
        r"^(" + "|".join(list_ok) + r")$",
        list_ok=list_ok,
    )

    check_format_features([event], "event", r"^\d{2}[PMF]$")

    type_liasse, nature, surface, event = (
        "NaN" if not isinstance(v, str) else v
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


def preprocess_batch(query: dict, nb_echos_max: int) -> dict:
    """
    Preprocesses a batch of data in a dictionary format for prediction.

    Args:
        query (dict): A dictionary containing the batch of data.
        nb_echos_max (int): The maximum number of echoes allowed.

    Returns:
        dict: A dictionary containing the preprocessed data ready for further
        processing.
    Raises:
        HTTPException: If the 'text_description' field is missing for any
            liasses in the batch, a HTTPException is raised with a 400
            status code and a detailed error message.
    """

    df = pd.DataFrame(query)
    df = df.apply(lambda x: x.str.strip())
    df = df.replace(["null", "", "NA", "NAN", "nan", "None"], np.nan)

    if df["text_description"].isna().any():
        matches = df.index[df["text_description"].isna()].to_list()
        raise HTTPException(
            status_code=400,
            detail=(
                "The text_description is missing for some liasses."
                f"See line(s): {*matches,}"
            ),
        )

    list_ok = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "G",
        "I",
        "L",
        "M",
        "N",
        "R",
        "S",
        "X",
        "Y",
        "Z",
    ]
    check_format_features(
        df["type_"].to_list(),
        "type_",
        r"^(" + "|".join(list_ok) + r")$",
        list_ok=list_ok,
    )

    check_format_features(df["nature"].to_list(), "nature", r"^\d{2}$")

    list_ok = ["1", "2", "3", "4"]
    check_format_features(
        df["surface"].to_list(),
        "surface",
        r"^(" + "|".join(list_ok) + r")$",
        list_ok=list_ok,
    )

    check_format_features(df["event"].to_list(), "event", r"^\d{2}[PMF]$")

    df = df.replace(np.nan, "NaN")

    df.rename(
        columns={
            "text_description": "TEXT_FEATURE",
            "type_": "AUTO",
            "nature": "NAT_SICORE",
            "surface": "SURF",
            "event": "EVT_SICORE",
        },
        inplace=True,
    )

    query = {
        "query": df.to_dict("list"),
        "k": nb_echos_max,
    }
    return query


def process_response(
    predictions: tuple,
    liasse_nb: int,
    nb_echos_max: int,
    prob_min: float,
    libs: dict,
):
    """
    Processes model predictions and generates response.

    Args:
        predictions (tuple): The model predictions as a tuple of two numpy
        arrays.
        nb_echos_max (int): The maximum number of echo predictions.
        prob_min (float): The minimum probability threshold for predictions.
        libs (dict): A dictionary containing mapping of codes to labels.

    Returns:
        response (dict): The processed response as a dictionary containing
        the predicted results.

    Raises:
        HTTPException: If the minimal probability requested is higher than
        the highest prediction probability of the model, a HTTPException
        is raised with a 400 status code and a detailed error message.
    """
    k = nb_echos_max
    if predictions[1][liasse_nb][-1] < prob_min:
        k = np.min(
            [
                np.argmax(
                    np.logical_not(predictions[1][liasse_nb] > prob_min)
                ),
                nb_echos_max,
            ]
        )

    output_dict = {
        rank_pred
        + 1: {
            "code": predictions[0][liasse_nb][rank_pred].replace(
                "__label__", ""
            ),
            "probabilite": float(predictions[1][liasse_nb][rank_pred]),
            "libelle": libs[
                predictions[0][liasse_nb][rank_pred].replace("__label__", "")
            ],
        }
        for rank_pred in range(k)
    }

    try:
        response = output_dict | {
            "IC": output_dict[1]["probabilite"]
            - float(predictions[1][liasse_nb][1])
        }
        return response
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=(
                "The minimal probability requested is "
                "higher than the highest prediction "
                "probability of the model."
            ),
        )


def check_format_features(
    values: list, feature: str, regex: str, list_ok: list = None
) -> None:
    """
    Check the format of values for a specific feature using regex pattern.

    Args:
        values (list): A list of values to be checked.
        feature (str): The name of the feature being checked.
        regex (str): The regex pattern used to check the format of values.
        list_ok (list, optional): A list of accepted values for the feature.

    Raises:
        HTTPException: If the format of any value in the list does not match
         the regex pattern, a HTTPException is raised with a
         400 status code and a detailed error message.
    """

    matches = []

    for i, value in enumerate(values):
        if isinstance(value, str):
            if not re.match(regex, value):
                matches.append(i)

    errors = {
        "type_": (
            "The format of type_liasse is incorrect. Accepted values are"
            f": {list_ok}. See line(s) : {*matches,}"
        ),
        "nature": (
            "The format of nature is incorrect. The nature is an "
            f"integer between 00 and 99. See line(s): {*matches,}"
        ),
        "surface": (
            "The format of surface is incorrect. Accepted values are: "
            f"{list_ok}. See line(s): {*matches,}"
        ),
        "event": (
            f"The format of event is incorrect. The event value is an "
            "integer between 00 and 99 plus the letter P, M or F. Example: "
            f"'01P'. See line(s): {*matches,}"
        ),
    }

    if matches:
        raise HTTPException(status_code=400, detail=errors[feature])
