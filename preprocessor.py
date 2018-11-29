import numpy as np
import pandas as pd
from utils import *
from constants import *
from pathlib import Path
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self, datafolder):
        self.datafolder = datafolder
        self.bus_data = None
        self.users_data = None
        self.review_data = None
        self.scaler = MinMaxScaler(feature_range=(1, 5))
        # self.scaler = StandardScaler()
        # self.nmlzr = Normalizer()

    # preprocess business info
    def preprocess_bus(self):
        bus_file = "business.csv"
        bus_data = getData(self.datafolder / bus_file, bus_features)
        # set business_id as the index
        bus_data.set_index('business_id', inplace=True)
        # turn categorical to numeric
        if 'categories' in bus_data.columns:
            one_hot_cat = bus_data['categories'].str.get_dummies(sep=", ")
            bus_data = bus_data.drop(columns='categories').join(one_hot_cat)
        if 'attributes_RestaurantsPriceRange2' in bus_data.columns:
            bus_data.fillna({"attributes_RestaurantsPriceRange2": 2}, inplace=True)

        # Rescaling the data
        # find numerical data columns
        num_cols = bus_data.columns[bus_data.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
        # rescaling the data
        bus_data[num_cols] = self.scaler.fit_transform(bus_data[num_cols])
        # fill nan value with mean
        bus_data.fillna(bus_data.mean(), inplace=True)
        if 'stars' in bus_data.columns:
            bus_data.rename(columns={'stars': 'bus_avg_stars'}, inplace=True)
        if 'review_count' in bus_data.columns:
            bus_data.rename(columns={'review_count': 'bus_review_count'}, inplace=True)

        bus_data.to_csv(self.datafolder / bus_dict_file)
        self.bus_data = bus_data
        print("Finished business.csv preprocess")

    # preprocess users info
    def preprocess_users(self):
        users_file = "users.csv"
        users_data = getData(self.datafolder / users_file, users_features)
        # set user_id as the index
        users_data.set_index('user_id', inplace=True)
        # fill nan value with mean
        users_data.fillna(users_data.mean(), inplace=True)
        # find numerical data columns
        num_cols = users_data.columns[users_data.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
        # rescaling the data
        users_data[num_cols] = self.scaler.fit_transform(users_data[num_cols])
        if 'average_stars' in users_data.columns:
            users_data.rename(columns={'average_stars': 'user_avg_stars'}, inplace=True)
        if 'review_count' in users_data.columns:
            users_data.rename(columns={'review_count': 'user_review_count'}, inplace=True)
        users_data.to_csv(self.datafolder / users_dict_file)
        self.users_data = users_data
        print("Finished users.csv preprocess")

    # # preprocess reviews
    def preprocess_reviews(self):
        # preprocess training data
        review_file = "train_reviews.csv"
        review_data = getData(self.datafolder / review_file, review_features)
        # replace user_id and business_id with related features
        user_train = review_data['user_id'].apply(lambda user_id: self.users_data.loc[user_id,:])
        bus_train = review_data['business_id'].apply(lambda bus_id: self.bus_data.loc[bus_id,:])
        # # join them all together
        review_data = pd.concat([user_train, bus_train, review_data[['stars']]], axis=1, sort=False)
        # # fill nan value with mean
        review_data.fillna(review_data.mean(), inplace=True)
        review_data.to_csv(self.datafolder / huge_train_data_file)
        self.review_data = review_data
        print("Finished train_reviews.csv preprocess")

    def preprocess_queries(self, filename, is_test=False):
        # preprocess validate_queries.csv, move this part to preprocessor latter
        bus_dict = getData(self.datafolder / bus_dict_file, index=0)
        users_dict = getData(self.datafolder / users_dict_file, index=0)
        test_data = None
        if is_test:
            test_data = getData(self.datafolder / filename)
        else:
            test_data = getData(self.datafolder / filename, index=0)
        user_test = test_data['user_id'].apply(lambda user_id: users_dict.loc[user_id,:])
        bus_test = test_data['business_id'].apply(lambda bus_id: bus_dict.loc[bus_id,:])
        X_test = pd.concat([user_test, bus_test], axis=1, sort=False)
        X_test.fillna(X_test.mean(), inplace=True)
        if not is_test:
            y_test = test_data['stars']
            return X_test, y_test
        else:
            return X_test
