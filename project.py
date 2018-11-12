import numpy as np
import pandas as pd
from pathlib import Path
from constants import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

################# GLOBAL FUNCTIONS TO BE MOVED LATER #################

#Move getData to a utils 

def getData(filepath, cols=None, index=None):
    return pd.read_csv(filepath, index_col=index, usecols=cols)

#Move RMSE to utils

def RMSE(y_pred, y_target):
    return np.sqrt(((y_pred - y_target) ** 2).mean())


################# Base Classifer Class Every Model Inherits From #################    
    
class Classifier(object) :
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()

################# Models #################

class Complete_Random_Model(Classifier):
    
    #Randomly Assigns a prediction value
    def __init__(self):
        pass
    def fit(self, X, y):
        pass
    def predict(self, X):
        y = np.random.choice([1.0,2.0,3.0,4.0,5.0], len(X.index))
        return y   
    
class Random_Prob_Model(Classifier):

    #Predicts the target yelp rating randomly based on the
    #probability distribution of the training data's ratings
    
    def __init__(self):
        self.y_probabilities = []
    
    def fit(self, X, y):
        #Disregard X, look at every value of y
        y_count = {1.0:0, 2.0:0, 3.0:0, 4.0:0, 5.0:0}
        
        #Count up how many of each y there is
        for i, rv in y.iteritems():
            y_count[rv] += 1 

        #Assign probabilities accordingly (loop later)
        self.y_probabilities.append(y_count[1.0] / len(y.index))
        self.y_probabilities.append(y_count[2.0] / len(y.index))
        self.y_probabilities.append(y_count[3.0] / len(y.index))
        self.y_probabilities.append(y_count[4.0] / len(y.index))
        self.y_probabilities.append(y_count[5.0] / len(y.index))
        
        return self
    
    def predict(self, X):
        
        #Check if fit occured
        if self.y_probabilities is None :
            raise Exception("A fit was never done...")
        
        #Randomly assign a value 1.0-5.0 with found distribution
        y = np.random.choice([1.0,2.0,3.0,4.0,5.0], len(X.index), self.y_probabilities)
        
        return y   
        

class SGDModel:
    def __init__(self):
        # self.sgdr = SGDClassifier(verbose=1, tol=1e-3, max_iter=1000, n_jobs=2)
        self.sgdr = SVC(verbose=True)

    def train(self, x_train, y_train):
        self.sgdr.fit(x_train, y_train)

    def predict(self, x_test):
        return self.sgdr.predict(x_test)

if __name__ == "__main__":
    datafolder = Path(data_path)
    #After we preprocess data and decide features we want, we will load in as numpy arrays instead.

    #Load in training data after preprocesing
    # train_data = getData(datafolder / train_data_file, None)
    # test_data = getData(datafolder / test_data_file, None)

    #Separate training features from prediction
        #In preprocess i renamed stars to given stars, FIX IF NECESSARY TO UR NAMES
    # train_data_X = train_data.drop(columns = ['given_stars'])
    # train_data_X = train_data_X.drop(train_data_X.columns[0],axis=1)
    # train_data_y = train_data['given_stars']

    #Separate testing garbage from predictions
    # test_data_X = test_data.drop(columns = ['stars'])
    # #This drops the first un-named column (REMOVE IF USING PREPROCESS DATA)
    # test_data_X = test_data_X.drop(test_data_X.columns[0],axis=1)
    # test_data_y = test_data['stars']

    #Do Complete Random Model
    # cf = Complete_Random_Model()
    # rpm_pred = cf.predict(test_data_X)
    # rpm_rmse = RMSE(rpm_pred, test_data_y.values)
    # print(rpm_rmse)

    #Do Random Probabilistic Model
    # cf = Random_Prob_Model()
    # cf.fit(train_data_X,train_data_y)
    # rpm_pred = cf.predict(test_data_X)
    # rpm_rmse = RMSE(rpm_pred, test_data_y.values)
    # print(rpm_rmse)

    #Should get ~2.07 or 2.08 for both
    #They both work just as horrible! Nice!

    # lets try SGDRegressor
    train_data = getData(datafolder / huge_train_data_file, index=0)
    X_train = train_data.drop(columns='stars')
    y_train = train_data['stars']

    # preprocess validate_queries.csv, move this part to preprocessor latter
    bus_dict = getData(datafolder / bus_dict_file, index=0)
    users_dict = getData(datafolder / users_dict_file, index=0)
    test_data = getData(datafolder / test_data_file, index=0)
    user_test = test_data['user_id'].apply(lambda user_id: users_dict.loc[user_id,:])
    bus_test = test_data['business_id'].apply(lambda bus_id: bus_dict.loc[bus_id,:])
    X_test = pd.concat([user_test, bus_test], axis=1, sort=False)
    X_test.fillna(X_test.mean(), inplace=True)
    y_target = test_data['stars']

    # Debug
    # print(train_data_X.head(1))
    # print(test_data_X.head(1))

    # init model
    sgdmodel = SGDModel()
    # train data
    sgdmodel.train(X_train.values, y_train.values)
    # predict
    y_pred = sgdmodel.predict(X_test.values)
    y_pred = np.around(y_pred)
    print(F"predicted y: {y_pred}")
    sdgr_rmse = RMSE(y_pred, y_target.values)
    print(F"This is SGDRegressor's RMSE: {sdgr_rmse}")

    #Print out rpm_pred with indexes if you want to submit this as our first submission.
