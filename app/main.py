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

model = get_model()

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

on_event("startup")
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



# Import Uvicorn & the necessary modules from FastAPI
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
# Import the PyCaret Regression module
import pycaret.regression as pycr
# Import other necessary packages
from dotenv import load_dotenv
import pandas as pd
import os
# Load the environment variables from the .env file into the application
load_dotenv() 
# Initialize the FastAPI application
app = FastAPI()
# Create a class to store the deployed model & use it for prediction
class Model:
    def __init__(self, modelname, bucketname):
        """
        To initalize the model
        modelname: Name of the model stored in the S3 bucket
        bucketname: Name of the S3 bucket
        """
        # Load the deployed model from Amazon S3
        self.model = pycr.load_model(
            modelname, 
            platform = 'aws', 
            authentication = { 'bucket' : bucketname }
        )
    
    def predict(self, data):
        """
        To use the loaded model to make predictions on the data
        data: Pandas DataFrame to perform predictions
        """
        # Return the column containing the predictions  
        # (i.e. 'Label') after converting it to a list
        predictions = pycr.predict_model(self.model, data=data).Label.to_list()
        return predictions
# Load the model that you had deployed earlier on S3. 
# Enter your respective bucket name in place of 'mlopsdvc170100035'
model = Model("lightgbm_deploy_1", "mlopsdvc170100035")
# Create the POST endpoint with path '/predict'
@app.post("/predict")
async def create_upload_file(file: UploadFile = File(...)):
    # Handle the file only if it is a CSV
    if file.filename.endswith(".csv"):
        # Create a temporary file with the same name as the uploaded 
        # CSV file to load the data into a pandas Dataframe
        with open(file.filename, "wb")as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)
        os.remove(file.filename)
        # Return a JSON object containing the model predictions
        return {
            "Labels": model.predict(data)
        }    
    else:
        # Raise a HTTP 400 Exception, indicating Bad Request 
        # (you can learn more about HTTP response status codes here)
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")
# Check if the environment variables for AWS access are available. 
# If not, exit the program
if os.getenv("AWS_ACCESS_KEY_ID") == None or os.getenv("AWS_SECRET_ACCESS_KEY") == None:
    exit(1)