from flask import Flask, request, jsonify
import importlib
import utils
from mysklearn.mypytable import MyPyTable
from mysklearn.myclassifiers import MyKNeighborsClassifier
import mysklearn.myeval as myeval

app = Flask(__name__)

# Load the mushroom dataset and prepare the data
mush_data = MyPyTable()
mush_data.load_from_file('/home/CPSC322finalProject/input_data/new_mushroom_cleaned.csv')

# Preprocess the data (normalization, splitting, etc.)
indexes_to_remove = utils.rand_inds(50000, 54034)
mush_data.drop_rows(indexes_to_remove)

df_mush = [row for row in mush_data.data]
season = [row[7] for row in df_mush]
stem_width = [row[5] for row in df_mush]
gill_color = [row[3] for row in df_mush]
cap_diameter = [row[0] for row in df_mush]

yummy_or_nah = [row[8] for row in df_mush]

# Normalize the data
season_norm = utils.normalize_data(season)
stem_width_norm = utils.normalize_data(stem_width)
gill_color_norm = utils.normalize_data(gill_color)
cap_diameter_norm = utils.normalize_data(cap_diameter)

data_zipped = list(zip(season_norm, stem_width_norm, gill_color_norm, cap_diameter_norm))
X_data = data_zipped

X_train, X_test, y_train, y_test = myeval.train_test_split(X_data, yummy_or_nah, test_size=0.33, random_state=1, shuffle=True)

# Train the KNN model
knn = MyKNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

@app.route('/')
def home():
    return "Mushroom Classification Web App"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the user (expecting JSON format)
    data = request.get_json()

    # Assuming the input data contains 'season', 'stem_width', 'gill_color', 'cap_diameter'
    input_data = [
        data['season'],
        data['stem_width'],
        data['gill_color'],
        data['cap_diameter']
    ]

    # Normalize input data just like we did with training data
    input_data_norm = [
        utils.normalize_data([input_data[0]]),
        utils.normalize_data([input_data[1]]),
        utils.normalize_data([input_data[2]]),
        utils.normalize_data([input_data[3]])
    ]

    # Make prediction using the trained classifier
    prediction = knn.predict([input_data_norm])

    # Return the result as JSON
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')


# http request
# http://127.0.0.1:5001/predict?season=0.5&stem_width=0.3&gill_color=0.7&cap_diameter=0.8
