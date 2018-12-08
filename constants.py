# File: constants.py

# File Related

# DATA FOLDER NAME
data_path = "data"

# DATA FILE NAMES
bus_file = "business.csv"
users_file = "users.csv"
review_file = "train_reviews.csv"
validate_data_file = "validate_queries.csv"
test_data_file = "test_queries.csv"

# BUSINESS AND USER DICT -> CSV (The interested features filled in / cleaned for each bus id and user id)
bus_dict_file = "bus_dict.csv"
users_dict_file = "users_dict.csv"

# Outputs after preprocessing
huge_train_data_file = "cleaned_train_review.csv" 
cleaned_validate_queries = "cleaned_validate_queries.csv"
cleaned_test_queries = "cleaned_test_queries.csv"

# Output after running project.py
submission_file = "submission.csv"


# Features related

#Business id is not numerical but oh well

bus_features_id = ["business_id"]

bus_features_numerical = ["stars", "review_count", "attributes_RestaurantsPriceRange2"]
                
bus_features_bool = ["attributes_BikeParking", "attributes_BusinessAcceptsCreditCards", "attributes_Caters",
                     "attributes_GoodForKids", "attributes_HasTV", "attributes_OutdoorSeating", "attributes_RestaurantsDelivery",  
                     "attributes_RestaurantsGoodForGroups", "attributes_RestaurantsReservations","attributes_RestaurantsTableService", 
                     "attributes_RestaurantsTakeOut", "attributes_WheelchairAccessible"]

# bus_features_drop = features to drop
bus_features_drop = bus_features_bool.copy()
bus_features_keep = ["attributes_BusinessAcceptsCreditCards", "attributes_RestaurantsGoodForGroups", "attributes_RestaurantsTakeOut",
                    "attributes_HasTV", "attributes_BikeParking" ] # Features to Keep
for feature in bus_features_keep:
    if feature in bus_features_drop:
        bus_features_drop.remove(feature)


# All user features we want are numerical
user_features_id = ["user_id"]
user_features_numerical = ["average_stars", "review_count", "useful"] 


# All 
review_features = ["user_id", "business_id", "stars"]

# Scale or not
scale = False