# File: constants.py

# files related

# data folder name
data_path = "data"
bus_dict_file = "bus_dict.csv"
users_dict_file = "users_dict.csv"

train_data_file = "train_reviews_w_bus_avg.csv" #This is only part of the training data with the buisness averages added on.
validate_data_file = "validate_queries.csv"
test_data_file = "test_queries.csv"

huge_train_data_file = "cleaned_train_review.csv" # 4.7GB training data... considering changing a encoding scheme for categories

submission_file = "submission.csv"

# possible useful features
# attributes_Ambience    parse this one
# attributes_BusinessParking    parse as well
# attributes_Caters    fill NaN with False
# attributes_RestaurantsDelivery    fill NaN with False
# attributes_RestaurantsPriceRange2    fill NaN with 2
# attributes_RestaurantReservations    fill NaN with False
# attributes_RestaurantTakeOut    fill NaN with True
# attributes_WiFi    fill NaN with no
bus_features = ["business_id", "stars", "review_count", "attributes_RestaurantsPriceRange2"] #  "categories",
users_features = ["user_id", "average_stars"] # , "review_count"
review_features = ["user_id", "business_id", "stars"]
