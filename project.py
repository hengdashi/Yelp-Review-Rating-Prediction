import math
import numpy as np
import pandas as pd
from utils import *
from constants import *
from pathlib import Path
import matplotlib.pyplot as plt
from preprocessor import Preprocessor

# Models
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
    
    
   #X_train = X_train.drop(columns=bus_features_bool) #IF U WANT TO DROP FEATURES THIS IS HOW
    X_train = X_train.values
    y_train = y_train.values
    

    
    #The indexes of the data frame arnt used so just ignore how it doesn't start w/ 0 
    X_val = getData(datafolder / cleaned_validate_queries, index = 0)   
    y_val = X_val['stars']
    X_val = X_val.drop(columns='stars')
   # X_val = X_val.drop(columns=bus_features_bool) #IF YOU WANT TO DROP FEATURES THIS IS HOW
    X_val = X_val.values  
    y_val = y_val.values


    X_test = getData( datafolder / cleaned_test_queries, index = 0) 
   # X_test = X_test.drop(columns=bus_features_bool) #IF U WANT TO DROP FEATURES THIS IS HOW
    X_test = X_test.values
    # we dont have y test for the uneducated
    # Use .values to convert to numpy
    
  


  # TODO  GRAPHS AND FEATURE ANALYSIS .....

    scaler = MinMaxScaler(feature_range=(1,5))
    if scale:
       X_train = scaler.fit_transform(X_train)
       X_val = scaler.transform(X_val)
       X_test = scaler.transform(X_test)
  
    # BEST MODEL SO FAR! 185, 4, .1 (1.0427 submitted, 1.0425 if attributes_Caters dropped
    
    for i in [185]:
        for j in [4]:
            for lr in [.1]: #.09 is a little better but not much
                gbd = GradientBoostingRegressor(n_estimators=i, learning_rate=lr, max_depth=j, random_state=0, loss='ls').fit(X_train, y_train)
                y_pred = gbd.predict(X_val)
                rmse = math.sqrt(mean_squared_error(y_val, y_pred))
                print("Validation RMSE is: {} {} {} {}".format(rmse, i , j, lr))
                y_pred = gbd.predict(X_test)
                submission = pd.DataFrame(y_pred, columns=['stars'])
                submission.to_csv(submission_file, index_label='index')
    print(gbd.feature_importances_)
    
    '''
    #Random Forest , 1.043 not the best
    for i in [150, 200, 250]:
        for j in [8]: 
            for k in [2,3,4]:
                rf = RandomForestRegressor(n_estimators=i, max_depth=j, random_state=0, min_samples_split = k).fit(X_train, y_train)
                y_pred = rf.predict(X_val)
                rmse = RMSE(y_val, y_pred)
                print("RF Validation RMSE is: {} {} {}".format(rmse, i , j))
        
    '''
    '''
    ###### Random Model ###### RMSE = ~2.1
    print("Random Model ...")
    clf = Complete_Random_Model()
    #Validation 
    y_pred = clf.predict(X_val)
    rmse = math.sqrt(mean_squared_error(y_val, y_pred))
    print("Random Validation RMSE is: {}".format(rmse))
    #Test 
    y_pred = clf.predict(X_test)
    submission = pd.DataFrame(y_pred, columns=['stars'])
    submission.to_csv(submission_file, index_label='index')
    print("Random Model FINISHED")
    
    
    ###### Random Probabilistic Model  ###### RMSE = ~1.7
    print("Random Probabilistic Model ...")
    clf = Random_Prob_Model()
    #Validation 
    clf.fit(X_train,y_train.tolist())
    y_pred = clf.predict(X_val)
    rmse = RMSE(y_pred, y_val)
    print(F"Random Probabilistic Validation RMSE is: {rmse}")
    #Test 
    y_pred = clf.predict(X_test)
    submission = pd.DataFrame(y_pred, columns=['stars'])
    submission.to_csv(submission_file, index_label='index')
    print("Random Probabilistic Model FINISHED")
    '''
    
    '''
    #validation rmse = 1.0522
    #Linear Regression
    print("Linear Regression Model ...")
    clf = LinearRegression()
    #Validation 
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_val)
    rmse = RMSE(y_pred, y_val)
    print(F"Linear Regression RMSE is: {rmse}")
    #Test 
    y_pred = clf.predict(X_test)
    submission = pd.DataFrame(y_pred, columns=['stars'])
    submission.to_csv(submission_file, index_label='index')
    print("Linear Regression FINISHED")
    '''
    
    '''
    #Regression with polynomial features 1.046 with polyfeatures(3)
    print("Linear Regression Model with polyfeatures ...")
    clf = LinearRegression()
    poly = PolynomialFeatures(3)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_val = poly.fit_transform(X_val)
    #Validation 
    clf.fit(X_poly_train,y_train)
    y_pred = clf.predict(X_poly_val)
    rmse = RMSE(y_pred, y_val)
    print(F"Linear Regression RMSE is: {rmse}")
    #Test 
    X_poly_test = poly.fit_transform(X_test)
    y_pred = clf.predict(X_poly_test)
    submission = pd.DataFrame(y_pred, columns=['stars'])
    submission.to_csv(submission_file, index_label='index')
    print("Linear Regression FINISHED")
    '''
    
    '''
    #Wont converge if the boolean features are included, remove them if using log regression
    print("Logistic Regression Model ...")
    clf = LogisticRegression()
    #Validation 
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_val)
    rmse = RMSE(y_pred, y_val)
    print(F"Log Regression RMSE is: {rmse}")
    #Test 
    y_pred = clf.predict(X_test)
    submission = pd.DataFrame(y_pred, columns=['stars'])
    submission.to_csv(submission_file, index_label='index')
    print("Linear Regression FINISHED")
    '''
    
    '''
    #1.048 ish with max depth of 7, 1.058 on submit
    #Decision Tree
    print("Decision Tree Model ...")
    print("Investingating Best Depth...")
    dt_rmse = []
    dt_max_depths = np.arange(1,21)
    for i in dt_max_depths:
        clf = DecisionTree(i)
        #Validation 
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_val)
        rmse = RMSE(y_val, y_pred)
        dt_rmse.append(rmse)
    best_k = np.argmin(dt_rmse) + 1
    print("Best depth is : {} with RMSE {}".format(best_k,dt_rmse[best_k-1]))
    plt.title('Validation RMSE vs Max Tree Depth')
    plt.xlabel('Max Tree Depth')
    plt.ylabel('Validation RMSE')
    plt.plot(dt_max_depths, dt_rmse, 'ko-')
    plt.show()
    #Refit to out submission
    clf = DecisionTree(best_k)
    #Validation 
    clf.fit(X_train,y_train)
    #Test 
    y_pred = clf.predict(X_test)
    submission = pd.DataFrame(y_pred, columns=['stars'])
    submission.to_csv(submission_file, index_label='index')
    print("Decision Tree Model FINISHED")    
    '''
    
    '''
    #KNN bad rmse like 1.2ish?
    #DO NOT USE SCALE AND THIS
    for w in ['uniform', 'distance']:
        for i in range(1,25):
            knn = neighbors.KNeighborsRegressor(i, w)
            clf = knn.fit(X_train,y_train)
            y_pred = clf.predict(X_val)
            rmse = RMSE(y_val, y_pred)
            print("i is {}, rmse is {}".format(i,rmse))
    '''
    
    '''
    # Neural Network
    regsr = MLPRegressor(verbose=True, max_iter=200, hidden_layer_sizes=100, learning_rate='adaptive', learning_rate_init=1e-4)
    regsr.fit(X_train, y_train)
    print("Finished training")
    y_pred = regsr.predict(X_val)
    print("Finished validation")
    sdgr_rmse = RMSE(y_pred, y_val)
    print(F"This is NN's RMSE: {sdgr_rmse}")
    y_pred = regsr.predict(X_test)
    print("Finished prediction")
    submission = pd.DataFrame(y_pred, columns=['stars'])
    submission.to_csv(submission_file, index_label='index')
    '''
    
    
    
### EXPERIMENTAL / OLD BELOW

    
    '''
    #rng = np.random.RandomState(1)
    #regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=13),
    #                      n_estimators=500, random_state=rng)
#
    #regr_2.fit(X_train,y_train)
    #y_pred = regr_2.predict(X_val)
    #rmse = RMSE(y_pred, y_val)
   # print(rmse)
    
    #for j in [500,2000,8000,99999]:
    #    clf_stump=DecisionTreeRegressor(max_features=None,max_leaf_nodes=j)
    #    for i in np.arange(1,26):
    #        bstlfy=AdaBoostRegressor(base_estimator=clf_stump,n_estimators=i)
    #    bstlfy.fit(X_train,y_train)
    #    y_pred = bstlfy.predict(X_val)
    #    rmse = RMSE(y_pred, y_val)
    #    print(rmse)
    
   # DTC = DecisionTreeRegressor(random_state = 11, max_features = "auto",max_depth = None)

    #ABC = AdaBoostRegressor(base_estimator = DTC)

    # run grid search
   # my_func = make_scorer(RMSE,greater_is_better=False)
   # grid_search_ABC = GridSearchCV(ABC, param_grid=param_grid, scoring =my_func)    
   # grid_search_ABC.fit(X_train,y_train)
   # print(grid_search_ABC.best_params_)
   '''
    
    
    # init model
    #regsr = Regressor()
    # train data
    #regsr.train(X_train, y_train)
    # plt.plot(regsr.getObject().loss_curve_)

    # used for minibatch SGD
    # perm = np.random.permutation(len(X_train))
    # X_train_random = X_train.values[perm]
    # y_train_random = y_train.values[perm]
    # chunk_size = 50
    # num_folds = math.ceil(X_train_random.shape[0] / chunk_size)
    # X_train_folds = np.asarray(np.array_split(X_train_random, num_folds))
    # y_train_folds = np.asarray(np.array_split(y_train_random, num_folds))
    # regsr.minibatch_train(X_train_folds, y_train_folds)

    # display the relative importance of each attribute
    # print(regsr.getObject().feature_importances_)

    #print("Finished training")
    # report(regsr.getparams())


        
    #clf = autosklearn.classification.AutoSklearnClassifier()
    #clf.fit(X_train, y_train)
    #y_pred = clf.predict(X_val)
    #rmse = RMSE(y_pred, y_val)
    #print("rmse is {}".format(rmse))
    
    
    #class Regressor(object):
    #def __init__(self):

        # self.regsr = SVC(verbose=True, shrinking=False, decision_function_shape='ovo', cache_size=500) # SVM

        # self.regsr = KNeighborsRegressor(n_neighbors=138, n_jobs=2) # KNN 1.052414

        # self.regsr = GaussianNB() # naive_bayes 1.22957

        # self.regsr = SGDRegressor(verbose=1, tol=1e-5, loss='huber') # 1.15333

        # self.regsr = LogisticRegressionCV(cv=5, solver='newton-cg', multi_class='multinomial', verbose=1, n_jobs=2) # Logistic 1.18866 fixed

        # Random Forest
        # self.regsr = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=2) # 1.1611

        # NN
        # params = {
        #             "activation": ['logistic', 'tanh', 'relu'],
        #             "solver": ['sgd', 'adam'],
        #             "alpha": [1e-2, 1e-3, 1e-4, 1e-5],
        #             "learning_rate": ['invscaling', 'adaptive'],
        #             "learning_rate_init": [1e-2, 1e-3, 1e-4, 1e-5]
        #          }
        #self.regsr = MLPRegressor(verbose=True, max_iter=200, hidden_layer_sizes=100) # -1.047
        # self.regsr = MLPRegressor(verbose=True, learning_rate='adaptive', learning_rate_init=1e-4)
        # self.regsr = GridSearchCV(self.regsr, params, cv=5, n_jobs=2, verbose=True)
        # self.regsr = RandomizedSearchCV(self.regsr, params, cv=5, n_jobs=2, verbose=True, n_iter=10)

        # self.kernel = AdditiveChi2Sampler(sample_steps=3)

        # feature importance
        # self.regsr = ExtraTreesClassifier(n_estimators=250, random_state=0)

    #def train(self, X_train, y_train):
        # X_train = self.kernel.fit_transform(X_train, y_train)
   #     self.regsr.fit(X_train, y_train)

    # only for SGD minibatch implementation
    #def minibatch_train(self, X_train, y_train):
    #    classes = np.arange(1.0, 6.0)
     #   for i in range(X_train.shape[0]):
     #       self.regsr.partial_fit(X_train[i], y_train[i], classes=classes)

    #def predict(self, X_test):
        # X_test = self.kernel.fit_transform(X_test)
    #    return self.regsr.predict(X_test)

    # only for NN
    #def getparams(self):
    #    return self.regsr.cv_results_

    #def getObject(self):
    #    return self.regsr