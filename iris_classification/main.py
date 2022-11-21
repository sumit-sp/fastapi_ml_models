from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
import pickle
import numpy as np

app = FastAPI()

#load trained ml model
with open('./rfc_iris.pkl', 'rb') as f:
    rfc_iris = pickle.load(f)

# request body structure using pydantic's BaseModel, it does data validation also
class iris_data(BaseModel):
    num1 : float
    num2 : float
    num3 : float
    num4 : float

@app.post('/iris_classifier')
async def load_iris_data(iris: iris_data):
    """
    data = iris.dict()
    x1 = data['num1']
    x2 = data['num2']
    x3 = data['num3']
    x4 = data['num4']

    prediction = rfc_iris.predict([[x1, x2, x3, x4]])
    """
    data = np.array([[iris.num1, iris.num2, iris.num3, iris.num4]])
    prediction = rfc_iris.predict(data)

    # Fastapi returns (respnse body) in JSON format only
    return {'iris_class': str(prediction[0])}
