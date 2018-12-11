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
        # Hengda what are you doing with these?
        # self.scaler = StandardScaler()
        # self.nmlzr = Normalizer()

    # PREPROCESS BUSINESS.CSV DATA
    def preprocess_bus(self):
        
        # EXTRACT THE DATA FROM BUISNESS.CSV THAT WE ARE INTERESTED IN
        
        bus_features = bus_features_id + bus_features_numerical + bus_features_bool + bus_features_cat
        
        bus_data = getData(self.datafolder / bus_file, bus_features)
        
        # PROCESS FEATURES OF INTEREST
       
        # SET BUISNESS ID AS INDEX
        
        bus_data.set_index('business_id', inplace=True)
        
        # FIX ANY MISSING VALUES OF NUMERICAL DATA
        if bus_features_numerical:
            # STARS AND REVIEW COUNT HAVE NO MISSING DATA
            # CHOOSING TWO (ABOUT AVERAGE) TO REPLACE 1268 MISSING PRICE RANGE VALUES
            if "attributes_RestaurantsPriceRange2" in bus_features_numerical and  "attributes_RestaurantsPriceRange2" in bus_data.columns:
                bus_data.fillna({"attributes_RestaurantsPriceRange2": 2}, inplace=True)
            # IF YOU ADD MORE NUMERICAL FEATURES YOU MIGHT NEED TO EDIT HERE DEPENDING ON IF IT HAS MISSING VALUES

            if "attributes_NoiseLevel" in bus_features_cat:
                bus_data.fillna({"attributes_NoiseLevel": "average"}, inplace=True)
                bus_data.replace("very_loud", 0, inplace=True)
                bus_data.replace("loud", 1, inplace=True)
                bus_data.replace("average", 2, inplace=True)
                bus_data.replace("quiet", 3, inplace=True)

        # LOOP THROUGH ALL BOOLEAN FEATURES AND REPLACE ANY MISSING VALUES WITH FALSE (0)
        for x in bus_features_bool:
            if x in bus_data.columns:
                bus_data.fillna({x: 0}, inplace=True)
        
        #COVERT ALL T/F TO 1/0
        if bus_features_bool:
            bus_data.replace(True, 1, inplace=True)
            bus_data.replace(False, 0, inplace=True)
                 
                
        # RENAME ANY CONFLICTING NAMES WITH OTHER FILES       
        if 'stars' in bus_data.columns:
            bus_data.rename(columns={'stars': 'bus_avg_stars'}, inplace=True)
            
        if 'review_count' in bus_data.columns:
            bus_data.rename(columns={'review_count': 'bus_review_count'}, inplace=True)

        # OUTPUT TO A CSV FILE TO SANITY CHECK
        bus_data.to_csv(self.datafolder / bus_dict_file)
        self.bus_data = bus_data
        
        # DONE
        print("Finished business.csv preprocess")

    # PREPROCESS USERS.CSV DATA
    def preprocess_users(self):
    
        # EXTRACT THE DATA FROM USERS.CSV THAT WE ARE INTERESTED IN
        users_features = user_features_id + user_features_numerical
        
        users_data = getData(self.datafolder / users_file, users_features)
        
        # PROCESS FEATURES OF INTEREST
        
        # SET USER_ID AS INDEX
        users_data.set_index('user_id', inplace=True)
        
        # NOT MISING ANY OF THE FOLLOWING DATA SO JUST RENAME CONFLICTS 
        
        if 'average_stars' in users_data.columns:
            users_data.rename(columns={'average_stars': 'user_avg_stars'}, inplace=True)
        if 'review_count' in users_data.columns:
            users_data.rename(columns={'review_count': 'user_review_count'}, inplace=True)
        if 'useful' in users_data.columns:
            users_data.rename(columns={'useful': 'user_useful'}, inplace=True)
        
        # OUTPUT TO A CSV FILE TO SANITY CHECK
        users_data.to_csv(self.datafolder / users_dict_file)
        self.users_data = users_data
        
        #DONE
        print("Finished users.csv preprocess")
        
        # fill nan value with mean
        #but there are none
        #users_data.fillna(users_data.mean(), inplace=True)
        # find numerical data columns
        #num_cols = users_data.columns[users_data.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
        # rescaling the data
        #users_data[num_cols] = self.scaler.fit_transform(users_data[num_cols])

    # PROCESS THE TRAINING REVIEWS
    def preprocess_reviews(self):
    
        review_data = getData(self.datafolder / review_file, review_features)
        
        # replace user_id and business_id with related features
        user_train = review_data['user_id'].apply(lambda user_id: self.users_data.loc[user_id,:])
        bus_train = review_data['business_id'].apply(lambda bus_id: self.bus_data.loc[bus_id,:])
        
        # join them all together
        review_data = pd.concat([user_train, bus_train, review_data[['stars']]], axis=1, sort=False)
       
        review_data.to_csv(self.datafolder / huge_train_data_file)
        self.review_data = review_data
        
        # No NaNs so everythign should be fine
        
        # DONE
        print("Finished train_reviews.csv preprocess")

    # PROCESS THE VALIDATION AND TEST QUERIES
    def preprocess_queries(self):
    
        # preprocess validate_queries.csv, move this part to preprocessor latter
        #bus_dict = getData(self.datafolder / bus_dict_file, index=0) BUS_DATA
        #users_dict = getData(self.datafolder / users_dict_file, index=0) USERS_DATA
        
        val_data = getData(self.datafolder / validate_data_file, index=0)
        test_data = getData(self.datafolder / test_data_file)
        
        user_val = val_data['user_id'].apply(lambda user_id: self.users_data.loc[user_id,:])
        bus_val = val_data['business_id'].apply(lambda bus_id: self.bus_data.loc[bus_id,:])
        
        user_test = test_data['user_id'].apply(lambda user_id: self.users_data.loc[user_id,:])
        bus_test = test_data['business_id'].apply(lambda bus_id: self.bus_data.loc[bus_id,:])
        
        val_proc = pd.concat([user_val, bus_val, val_data["stars"]], axis=1, sort=False)
        test_proc = pd.concat([user_test, bus_test], axis=1, sort=False)
        
        #Will fill nans in the project file instead
        #val_proc.fillna(val_proc.mean(), inplace=True)
        
        val_proc.to_csv(self.datafolder / cleaned_validate_queries)
        test_proc.to_csv(self.datafolder / cleaned_test_queries)
        
        # DONE
        print("Finished validation and test queries processing")
        
        #if not is_test:
        #    y_test = test_data['stars']
        #    return X_test, y_test
        ##else:
        #   return X_test
