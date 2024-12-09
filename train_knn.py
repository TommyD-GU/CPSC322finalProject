import pickle
from mysklearn.myknnclassifier import MyKNeighborsClassifier
from mysklearn.mypytable import MyPyTable
import utils
from mysklearn import myeval

# Load and preprocess data
mush_data = MyPyTable()
mush_data.load_from_file('/home/CPSC322finalProject/input_data/new_mushroom_cleaned.csv')

# Data preprocessing (similar to what you did before)
indexes_to_remove = utils.rand_inds(50000, 54034)
mush_data.drop_rows(indexes_to_remove)

df_mush = [row for row in mush_data.data]
season = [row[7] for row in df_mush]
stem_width = [row[5] for row in df_mush]
gill_color = [row[3] for row in df_mush]
cap_diameter = [row[0] for row in df_mush]
yummy_or_nah = [row[8] for row in df_mush]

# Normalize data
season_norm = utils.normalize_data(season)
stem_width_norm = utils.normalize_data(stem_width)
gill_color_norm = utils.normalize_data(gill_color)
cap_diameter_norm = utils.normalize_data(cap_diameter)

data_zipped = list(zip(season_norm, stem_width_norm, gill_color_norm, cap_diameter_norm))
X_data = data_zipped
X_train, X_test, y_train, y_test = myeval.train_test_split(X_data, yummy_or_nah, test_size=0.33, random_state=1, shuffle=True)

# Train KNN model
knn = MyKNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Save the trained KNN model to a pickle file
with open("knn_model.p", "wb") as outfile:
    pickle.dump(knn, outfile)

print("KNN model has been trained and saved as 'knn_model.p'")
