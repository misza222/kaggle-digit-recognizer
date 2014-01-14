

from sklearn.linear_model import LinearRegression
import numpy as np

def LR_learn(Xtrain, ytrain):
  """Return a linear model representation that has learned on Xtrain and ytrain data."""
  lr = LinearRegression()
  lr.fit(Xtrain, ytrain)
  return lr
  
def LR_predict(lrf, data):
  """Return a prediction based on a linear model."""
  return np.around(lrf.predict(data))
