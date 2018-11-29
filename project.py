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
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
# from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.kernel_approximation import *

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

################# Models #################
############ Base Classifer Class Every Model Inherits From ############
class Classifier(object):
    def __init__(self):
        # self.clsfr = DecisionTreeClassifier(max_features='auto') # Decision Tree 1.22183

        # self.clsfr = SVC(verbose=True, shrinking=False, decision_function_shape='ovo', cache_size=500) # SVM

        # self.clsfr = KNeighborsClassifier(n_neighbors = 92, n_jobs=2) # KNN 1.20478

        # self.clsfr = GaussianNB() # naive_bayes 1.22957

        # self.clsfr = SGDClassifier(verbose=1, n_jobs=2, tol=1e-5) # linear svm SGD 1.22123

        # self.clsfr = SGDClassifier(verbose=1, n_jobs=2, tol=1e-5, loss='log') # log SGD 1.24033

        # self.clsfr = SGDClassifier(verbose=1, n_jobs=1, tol=1e-5, loss='modified_huber') # 1.15333

        # self.clsfr = SGDClassifier(verbose=1, n_jobs=2, tol=1e-5, loss='squared_hinge') # 1.37178

        # self.clsfr = LogisticRegressionCV(cv=5, solver='newton-cg', multi_class='multinomial', verbose=1, n_jobs=2) # Logistic 1.18866 fixed

        # Random Forest
        # self.clsfr = RandomForestClassifier(n_estimators=500, verbose=1, n_jobs=2) # 1.21995

        # NN
        # params = {
        #             "activation": ['logistic', 'tanh', 'relu'],
        #             "solver": ['sgd', 'adam'],
        #             "alpha": [1e-2, 1e-3, 1e-4, 1e-5],
        #             "learning_rate": ['invscaling', 'adaptive'],
        #             "learning_rate_init": [1e-2, 1e-3, 1e-4, 1e-5]
        #          }
        self.clsfr = MLPClassifier(verbose=True, max_iter=200, hidden_layer_sizes=100, batch_size=400, tol=1e-4, solver='adam', learning_rate='adaptive', learning_rate_init=2e-5)
        # self.clsfr = GridSearchCV(self.clsfr, params, cv=5, n_jobs=2, verbose=True)
        # self.clsfr = RandomizedSearchCV(self.clsfr, params, cv=5, n_jobs=2, verbose=True, n_iter=10)

        # self.kernel = AdditiveChi2Sampler(sample_steps=3)

        # feature importance
        # self.clsfr = ExtraTreesClassifier()

    def train(self, X_train, y_train):
        # X_train = self.kernel.fit_transform(X_train, y_train)
        self.clsfr.fit(X_train, y_train)

    # only for SGD minibatch implementation
    def minibatch_train(self, X_train, y_train):
        classes = np.arange(1.0, 6.0)
        for i in range(X_train.shape[0]):
            self.clsfr.partial_fit(X_train[i], y_train[i], classes=classes)

    def predict(self, X_test):
        # X_test = self.kernel.fit_transform(X_test)
        return self.clsfr.predict(X_test)

    # only for NN
    def getparams(self):
        return self.clsfr.cv_results_

    def getObject(self):
        return self.clsfr

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
        if self.y_probabilities is None:
            raise Exception("A fit was never done...")

        #Randomly assign a value 1.0-5.0 with found distribution
        y = np.random.choice([1.0,2.0,3.0,4.0,5.0], len(X.index), self.y_probabilities)

        return y

if __name__ == "__main__":
    datafolder = Path(data_path)

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
    plt.plot(clsfr.getObject().loss_curve_)

    # used for minibatch SGD
    # perm = np.random.permutation(len(X_train.values))
    # X_train_random = X_train.values[perm]
    # y_train_random = y_train.values[perm]
    # chunk_size = 50
    # num_folds = math.ceil(X_train_random.shape[0] / chunk_size)
    # X_train_folds = np.asarray(np.array_split(X_train_random, num_folds))
    # y_train_folds = np.asarray(np.array_split(y_train_random, num_folds))
    # clsfr.minibatch_train(X_train_folds, y_train_folds)

    # display the relative importance of each attribute
    # print(clsfr.getObject().feature_importances_)

    print("Finished training")
    # report(clsfr.getparams())

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

    plt.show()
