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

from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.kernel_approximation import *

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

################# Models #################
############ Base Classifer Class Every Model Inherits From ############
class Regressor(object):
    def __init__(self):
        # self.regsr = DecisionTreeRegressor(max_features='auto') # Decision Tree 1.22183

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
        self.regsr = MLPRegressor(verbose=True, max_iter=200, hidden_layer_sizes=100) # -1.047
        # self.regsr = MLPRegressor(verbose=True, learning_rate='adaptive', learning_rate_init=1e-4)
        # self.regsr = GridSearchCV(self.regsr, params, cv=5, n_jobs=2, verbose=True)
        # self.regsr = RandomizedSearchCV(self.regsr, params, cv=5, n_jobs=2, verbose=True, n_iter=10)

        # self.kernel = AdditiveChi2Sampler(sample_steps=3)

        # feature importance
        # self.regsr = ExtraTreesClassifier(n_estimators=250, random_state=0)

    def train(self, X_train, y_train):
        # X_train = self.kernel.fit_transform(X_train, y_train)
        self.regsr.fit(X_train, y_train)

    # only for SGD minibatch implementation
    def minibatch_train(self, X_train, y_train):
        classes = np.arange(1.0, 6.0)
        for i in range(X_train.shape[0]):
            self.regsr.partial_fit(X_train[i], y_train[i], classes=classes)

    def predict(self, X_test):
        # X_test = self.kernel.fit_transform(X_test)
        return self.regsr.predict(X_test)

    # only for NN
    def getparams(self):
        return self.regsr.cv_results_

    def getObject(self):
        return self.regsr


if __name__ == "__main__":
    datafolder = Path(data_path)

    train_data = getData(datafolder / huge_train_data_file, index=0)
    X_train = train_data.drop(columns='stars')
    y_train = train_data['stars']

    # Debug
    # print(train_data_X.head(1))
    # print(test_data_X.head(1))

    # init model
    regsr = Regressor()
    # train data
    regsr.train(X_train.values, y_train.values)
    # plt.plot(regsr.getObject().loss_curve_)

    # used for minibatch SGD
    # perm = np.random.permutation(len(X_train.values))
    # X_train_random = X_train.values[perm]
    # y_train_random = y_train.values[perm]
    # chunk_size = 50
    # num_folds = math.ceil(X_train_random.shape[0] / chunk_size)
    # X_train_folds = np.asarray(np.array_split(X_train_random, num_folds))
    # y_train_folds = np.asarray(np.array_split(y_train_random, num_folds))
    # regsr.minibatch_train(X_train_folds, y_train_folds)

    # display the relative importance of each attribute
    # print(regsr.getObject().feature_importances_)

    print("Finished training")
    # report(regsr.getparams())

    # predict
    preprocessor = Preprocessor(datafolder)
    X_test, y_test = preprocessor.preprocess_queries(validate_data_file)
    y_pred = regsr.predict(X_test.values)
    print("Finished validation")

    sdgr_rmse = RMSE(y_pred, y_test.values)
    print(F"This is Regressor's RMSE: {sdgr_rmse}")
    X_test = preprocessor.preprocess_queries(test_data_file, is_test=True)
    y_pred = regsr.predict(X_test.values)
    print("Finished prediction")

    submission = pd.DataFrame(y_pred, columns=['stars'])
    submission.to_csv(submission_file, index_label='index')

    # plt.show()
