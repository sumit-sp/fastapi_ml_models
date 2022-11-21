import numpy as np
import pandas as pd

from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pickle

X, y = load_iris(return_X_y= True, as_frame= True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.75, random_state= 42, shuffle= True)

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

with open('./rfc_iris.pkl', 'wb') as f:
    pickle.dump('rfc', f)
