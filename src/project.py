import math
import numpy as np
import pandas as pd
from utils import *
from constants import *
from pathlib import Path
import matplotlib.pyplot as plt
from preprocessor import Preprocessor

# Models (Didn't have time to clean up imports)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn import linear_model
from sklearn import neighbors

from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.kernel_approximation import *

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# import autosklearn.regression
# from tpot import TPOTRegressor

#from yellowbrick.features.importances import FeatureImportances

############ Base Classifier/Regressors Class Every Model Will Inherit From ############

class Classifier(object):
    def fit(self, X_train, y_train):
        raise NotImplementedError()
    def predict(self, X_test):
        raise NotImplementedError()

class Regressor(object):
    def fit(self, X_train, y_train):
        raise NotImplementedError()
        
    def predict(self, X_test):
        raise NotImplementedError()
       
############ Models ############

class Complete_Random_Model(Classifier):
    #Random Classifier is bad, just a baseline to see HOW bad RMSE can be
    #Randomly Assigns a prediction value
    def __init__(self):
        pass
    def fit(self, X_train, y_train):
        pass
        
    def predict(self, X_test):
        return np.random.choice([1.0,2.0,3.0,4.0,5.0], X_test.shape[0])
 
class Random_Prob_Model(Classifier):
    #Just as bad as completely random
    #Predicts the target yelp rating randomly based on the
    #probability distribution of the training data's ratings
    def __init__(self):
        self.y_probabilities = []
    
    def fit(self, X_train, y_train):
        #Disregard X, look at every value of y
        y_count = {1.0:0, 2.0:0, 3.0:0, 4.0:0, 5.0:0}

        #Count up how many of each y there is
        for r in y_train:
            y_count[r] += 1

        #Assign probabilities accordingly (loop later)
        self.y_probabilities.append(float(y_count[1.0]) / len(y_train))
        self.y_probabilities.append(float(y_count[2.0]) / len(y_train))
        self.y_probabilities.append(float(y_count[3.0]) / len(y_train))
        self.y_probabilities.append(float(y_count[4.0]) / len(y_train))
        self.y_probabilities.append(float(y_count[5.0]) / len(y_train))

    def predict(self, X_test):
        #Check if fit occurred
        if self.y_probabilities is None :
            raise Exception("A fit was never carried out...")

        #Randomly assign a value 1.0-5.0 with found distribution
        y = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], X_test.shape[0], p=self.y_probabilities)

        return y

class DecisionTree(Regressor):
    def __init__(self, depth):
        self.regsr = DecisionTreeRegressor(max_depth = depth)

    def fit(self, X_train, y_train):
        self.regsr.fit(X_train, y_train)

    def predict(self, X_test):
        return self.regsr.predict(X_test)

class LinearRegression(Regressor):
    def __init__(self):
        self.regsr = linear_model.LinearRegression()

    def fit(self, X_train, y_train):
        self.regsr.fit(X_train, y_train)

    def predict(self, X_test):
        return self.regsr.predict(X_test)
    
class LogisticRegression(Classifier):
    def __init__(self):
        self.regsr = LogisticRegressionCV(cv=5, solver='newton-cg', multi_class='multinomial', verbose=1, n_jobs=2) # Logistic 1.18866 fixed

    def fit(self, X_train, y_train):
        self.regsr.fit(X_train, y_train)        
        
    def predict(self, X_test):
        return self.regsr.predict(X_test)   


if __name__ == "__main__":
    # Load in Training and Testing Data
    datafolder = Path(data_path)
    X_train = getData(datafolder / huge_train_data_file, index = 0) 
    y_train = X_train['stars']
    X_train = X_train.drop(columns="stars")
    #Visualize features 
    #YOU NEED TO PIP ISNTALL YELLOWBRICK TO USE THIS!
    #feat = plt.figure()
    #ax = feat.add_subplot()
    #g = FeatureImportances(GradientBoostingRegressor(),ax=ax)
    #g.fit(X_train,y_train)
    #g.poof()
    # X_train = X_train.drop(columns=bus_features_bool) # TO DROP FEATURES FEED AN ARRAY IN WITH THE FEATURES U WANT TO DROP #ATTRIBUTE CATERING DROPPED IN PREPROCESS INSTEAD
    X_train = X_train.values
    y_train = y_train.values


    # The indexes of the data frame arnt used so just ignore how it doesn't start w/ 0 
    X_val = getData(datafolder / cleaned_validate_queries, index = 0)   
    y_val = X_val['stars']
    X_val = X_val.drop(columns='stars')
    # X_val = X_val.drop(columns=bus_features_bool) # TO DROP FEATURES FEED AN ARRAY IN WITH THE FEATURES U WANT TO DROP #ATTRIBUTE CATERING DROPPED IN PREPROCESS INSTEAD
    X_val = X_val.values
    y_val = y_val.values


    X_test = getData( datafolder / cleaned_test_queries, index = 0) 
    # X_test = X_test.drop(columns=bus_features_bool) # TO DROP FEATURES FEED AN ARRAY IN WITH THE FEATURES U WANT TO DROP #ATTRIBUTE CATERING DROPPED IN PREPROCESS INSTEAD
    X_test = X_test.values

    # Used .values to convert to numpy

    scaler = MinMaxScaler(feature_range=(1,5))
    if scale:
       X_train = scaler.fit_transform(X_train)
       X_val = scaler.transform(X_val)
       X_test = scaler.transform(X_test)
       
    print("Starting with our best model...")
    # BEST MODEL SO FAR! 185, 4, .1 (1.0427 submitted, 1.0425 if attributes_Caters dropped
    # validation 1.04234 added noise level and removed catering
    # Note the loop is there for us to tune parameters ( we optimized one at a time as explained in report )
    # Using 145, 4, .1 for submission and 125, 4, .1
    for i in [146]:
        for j in [4]:
            for lr in [.1]: #.09 is a little better but not much
                gbd = GradientBoostingRegressor(n_estimators=i, learning_rate=lr, max_depth=j, random_state=0, max_features="auto").fit(X_train, y_train)
                y_pred = gbd.predict(X_val)
                rmse = RMSE(y_val, y_pred)
                print("GBR - Validation RMSE is: {} {} {} {}".format(rmse, i , j, lr))
    y_pred = gbd.predict(X_test)
    submission = pd.DataFrame(y_pred, columns=['stars'])
    submission.to_csv(submission_file, index_label='index')


    ###### Random Model ###### RMSE = ~2.1
    # clf = Complete_Random_Model()
    # #Validation 
    # y_pred = clf.predict(X_val)
    # rmse = math.sqrt(mean_squared_error(y_val, y_pred))
    # print("Random - Validation RMSE is: {}".format(rmse))
    #Test 
    #y_pred = clf.predict(X_test)
    #submission = pd.DataFrame(y_pred, columns=['stars'])
    #submission.to_csv(submission_file, index_label='index')


    ###### Random Probabilistic Model  ###### RMSE = ~1.7
    # clf = Random_Prob_Model()
    # #Validation 
    # clf.fit(X_train,y_train.tolist())
    # y_pred = clf.predict(X_val)
    # rmse = RMSE(y_pred, y_val)
    # print(F"Random Probabilistic - Validation RMSE is: {rmse}")
    #Test 
    #y_pred = clf.predict(X_test)
    #submission = pd.DataFrame(y_pred, columns=['stars'])
    #submission.to_csv(submission_file, index_label='index')


    #Linear Regression
    # clf = LinearRegression()
    # #Validation 
    # clf.fit(X_train,y_train)
    # y_pred = clf.predict(X_val)
    # rmse = RMSE(y_pred, y_val)
    # print(F"Linear Regression - Validation RMSE is: {rmse}")
    #Test 
    #y_pred = clf.predict(X_test)
    #submission = pd.DataFrame(y_pred, columns=['stars'])
    #submission.to_csv(submission_file, index_label='index')


    #Regression with polynomial features 1.046 with polyfeatures(3)
    #Tuned, best was 2 or 3 but this isnt close to our best so not redoing
    # clf = LinearRegression()
    # poly = PolynomialFeatures(2)
    # X_poly_train = poly.fit_transform(X_train)
    # X_poly_val = poly.fit_transform(X_val)
    # #Validation 
    # clf.fit(X_poly_train,y_train)
    # y_pred = clf.predict(X_poly_val)
    # rmse = RMSE(y_pred, y_val)
    # print(F"Linear Regression with poly features 3 -  RMSE is: {rmse}")
    #Test 
    #X_poly_test = poly.fit_transform(X_test)
    #y_pred = clf.predict(X_poly_test)
    #submission = pd.DataFrame(y_pred, columns=['stars'])
    #submission.to_csv(submission_file, index_label='index')


    #COMMENTING OUT BECAUSE NOT GOING TO CONVERGE
    #Wont converge if the boolean features are included, remove them if using
    #print("Logistic Regression Model ...")
    #clf = LogisticRegression()
    #Validation 
    #clf.fit(X_train,y_train)
    #y_pred = clf.predict(X_val)
    #rmse = RMSE(y_pred, y_val)
    #print(F"Log Regression RMSE is: {rmse}")
    #Test 
    #y_pred = clf.predict(X_test)
    #submission = pd.DataFrame(y_pred, columns=['stars'])
    #submission.to_csv(submission_file, index_label='index')
    #print("Linear Regression FINISHED")


    #1.048 ish with max depth of 7, 1.058 on submit
    #Decision Tree
    # print("Decision Tree Model ...")
    # print("Investigating Best Depth...")
    # dt_rmse = []
    # dt_max_depths = np.arange(1,21)
    # for i in dt_max_depths:
    #     clf = DecisionTree(i)
    #     #Validation 
    #     clf.fit(X_train,y_train)
    #     y_pred = clf.predict(X_val)
    #     rmse = RMSE(y_val, y_pred)
    #     dt_rmse.append(rmse)
    # best_d = np.argmin(dt_rmse) + 1
    # print("Best depth is : {} with RMSE {}".format(best_d,dt_rmse[best_d-1]))
    # plt.title('Validation RMSE vs Max Tree Depth')
    # plt.xlabel('Max Tree Depth')
    # plt.ylabel('Validation RMSE')
    # plt.plot(dt_max_depths, dt_rmse, 'ko-')
    # plt.show()
    #Refit for submission
    #clf = DecisionTree(best_d)
    #Validation 
    #clf.fit(X_train,y_train)
    #Test 
    #y_pred = clf.predict(X_test)
    #submission = pd.DataFrame(y_pred, columns=['stars'])
    #submission.to_csv(submission_file, index_label='index')

    #KNN 
    print("KNN COMMENTED OUT BECAUSE IT TAKES TIME")
    '''
    print("KNN ...")
    print("Investigating Best K...")
    k_rmses = []
    k_vals = np.arange(10,25)
    for w in ['distance']: #'uniform' wasn't better
        for i in k_vals:
            knn = neighbors.KNeighborsRegressor(i, w)
            clf = knn.fit(X_train,y_train)
            y_pred = clf.predict(X_val)
            rmse = RMSE(y_val, y_pred)
            k_rmses.append(rmse)
    best_k = np.argmin(k_rmses) + 1
    print("Best k is : {} with RMSE {}".format(best_k,k_rmses[best_k-1]))
    plt.title('Validation RMSE vs K val used for KNN')
    plt.xlabel('K')
    plt.ylabel('Validation RMSE')
    plt.plot(k_vals, k_rmses, 'ko-')
    plt.show()
    '''


    # Neural Network
    # regsr = MLPRegressor(verbose=False, max_iter=200, hidden_layer_sizes=100, learning_rate='adaptive', learning_rate_init=1e-4)
    # regsr.fit(X_train, y_train)
    # y_pred = regsr.predict(X_val)
    # print("Finished validation")
    # sdgr_rmse = RMSE(y_pred, y_val)
    # print(F"NN - Validation RMSE: {sdgr_rmse}")
    # y_pred = regsr.predict(X_test)
    #print("Finished prediction")
    #submission = pd.DataFrame(y_pred, columns=['stars'])
    #submission.to_csv(submission_file, index_label='index')


    #Random Forest
    #Uncomment to do a very slow search over 6 hyper parameters, none of which beat GBR
    print("RANDOM FOREST COMMENTED B/C IT TAKES AWHILE - SKIPPING")
    '''
    for i in [150, 200, 250]:
        for j in [8]: 
            for k in [2,3,4]:
                rf = RandomForestRegressor(n_estimators=i, max_depth=j, random_state=0, min_samples_split = k).fit(X_train, y_train)
                y_pred = rf.predict(X_val)
                rmse = RMSE(y_val, y_pred)
                print("RF Validation RMSE is: {} {} {} {}".format(rmse, i , j, k))
    '''
