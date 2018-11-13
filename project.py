import numpy as np
import pandas as pd
from utils import *
from constants import *
from pathlib import Path
from preprocessor import Preprocessor

# Models
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


################# Models #################
############ Base Classifer Class Every Model Inherits From ############
class Classifier(object):
    def __init__(self):
        # self.clsfr = DecisionTreeClassifier(max_features='auto') # Decision Tree 1.22183

        # self.clsfr = SVC(verbose=True, shrinking=False, decision_function_shape='ovo', cache_size=500) # SVM
        
        # self.clsfr = KNeighborsClassifier(n_neighbors = 92) # KNN 1.20478

        # self.clsfr = GaussianNB() # naive_bayes 1.22957

        # self.clsfr = SGDClassifier(verbose=1, n_jobs=2, tol=1e-5) # linear svm SGD 1.22123

        # self.clsfr = SGDClassifier(verbose=1, n_jobs=2, tol=1e-5, loss='log') # log SGD 1.24033

        # self.clsfr = SGDClassifier(verbose=1, n_jobs=2, tol=1e-5, loss='modified_huber') # 1.15333

        # self.clsfr = SGDClassifier(verbose=1, n_jobs=2, tol=1e-5, loss='squared_hinge') # 1.37178

        # self.clsfr = LogisticRegression(solver='newton-cg', multi_class='multinomial', verbose=1, n_jobs=3) # Logistic 1.18866 fixed
        # Random Forest
        self.clsfr = RandomForestClassifier(n_estimators=500, verbose=1, n_jobs=2) # 1.21995

    def train(self, X_train, y_train):
        self.clsfr.fit(X_train, y_train)

    def predict(self, X_test):
        return self.clsfr.predict(X_test)

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
    #Print out rpm_pred with indexes if you want to submit this as our first submission.

    # lets try SGDRegressor
    train_data = getData(datafolder / huge_train_data_file, index=0)
    X_train = train_data.drop(columns='stars')
    y_train = train_data['stars']

    # Debug
    # print(train_data_X.head(1))
    # print(test_data_X.head(1))

    # init model
    clsfr = Classifier()
    # train data
    clsfr.train(X_train.values, y_train.values)
    print("Finished training")
    # predict
    preprocessor = Preprocessor(datafolder)
    X_test, y_test = preprocessor.preprocess_queries(validate_data_file)
    y_pred = clsfr.predict(X_test.values)
    print("Finished validation")
    y_pred = np.around(y_pred)
    sdgr_rmse = RMSE(y_pred, y_test.values)
    print(F"This is Classifier's RMSE: {sdgr_rmse}")
    X_test = preprocessor.preprocess_queries(test_data_file, is_test=True)
    y_pred = clsfr.predict(X_test.values)
    print("Finished prediction")
    y_pred = np.around(y_pred)
    submission = pd.DataFrame(y_pred, columns=['stars'])
    submission.to_csv(submission_file, index_label='index')
