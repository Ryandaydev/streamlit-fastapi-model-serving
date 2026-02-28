"""Iris API"""

from fastapi import Depends, FastAPI, HTTPException, Query
import onnxruntime as rt
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel


api_description = "This API provides the inference for a machine learning model"


# Load the ONNX model
sess = rt.InferenceSession("rf_iris.onnx", providers=["CPUExecutionProvider"])

# Get the input and output names of the model
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

# FastAPI constructor with additional details added for OpenAPI Specification
app = FastAPI(
    description=api_description,
    title="IRIS API",
    version="0.1",
)

# Input Pydantic model (data validation)
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Output Pydantic model (prediction result)
class PredictionOutput(BaseModel):
    predicted_label: int


# Define the prediction route
@app.post("/predict/", response_model=PredictionOutput)
def predict(features: IrisFeatures):
    # Convert Pydantic model to NumPy array
    input_data = np.array([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]], dtype=np.float32)
    
    # Perform ONNX inference
    pred_onx = sess.run([label_name], {input_name: input_data})[0]
    
    # Return prediction as a Pydantic response model
    return PredictionOutput(predicted_label=int(pred_onx[0]))