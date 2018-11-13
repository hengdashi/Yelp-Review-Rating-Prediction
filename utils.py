import numpy as np
import pandas as pd
from pathlib import Path

################# GLOBAL FUNCTIONS TO BE MOVED LATER #################

def getData(filepath, cols=None, index=None):
    return pd.read_csv(filepath, index_col=index, usecols=cols)


def RMSE(y_pred, y_target):
    return np.sqrt(((y_pred - y_target) ** 2).mean())