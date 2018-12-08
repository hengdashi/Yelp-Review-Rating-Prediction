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

# Utility function to report best scores
#def report(results, n_top=3):
    #for i in range(1, n_top + 1):
        #candidates = np.flatnonzero(results['rank_test_score'] == i)
        #for candidate in candidates:
            #print(F"Model with rank: {i}")
            #print(F"Mean validation score: {results['mean_test_score'][candidate]} (std: {results['std_test_score'][candidate]})")
            #print(F"Parameters: {results['params'][candidate]}")
            #print("")
