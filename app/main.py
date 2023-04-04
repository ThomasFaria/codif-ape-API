import os
import warnings
from typing import List

import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI

codification_ape_app = FastAPI()


