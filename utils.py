import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error

################# GLOBAL FUNCTIONS TO BE MOVED LATER #################

def getData(filepath, cols=None, index=None):
    return pd.read_csv(filepath, index_col=index, usecols=cols)

def RMSE(y_pred, y_target):
    return math.sqrt(mean_squared_error(y_pred,y_target))

