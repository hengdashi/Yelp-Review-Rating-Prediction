# File: constants.py

# files related

# data folder name
data_path = "data"
bus_dict_file = "bus_dict.csv"
users_dict_file = "users_dict.csv"

train_data_file = "train_reviews_w_bus_avg.csv" #This is only part of the training data with the buisness averages added on.
test_data_file = "validate_queries.csv"

huge_train_data_file = "cleaned_train_review.csv" # 4.7GB training data... considering changing a encoding scheme for categories

bus_features = ["business_id", "stars", "categories", "review_count"]
users_features = ["user_id", "average_stars", "review_count"]
review_features = ["user_id", "business_id", "stars"]
