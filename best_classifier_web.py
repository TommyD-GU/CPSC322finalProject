from flask import Flask, request, jsonify
from mysklearn.myclassifiers import MyKNeighborsClassifier  # Import your custom KNN classifier

app = Flask(__name__)

# Load your training data (replace with actual data loading)
# Example: replace this with the actual training dataset and labels
X_train = [
    [1.0, 2.1, 3.2, 4.1],  # Example training instances
    [1.5, 2.0, 3.5, 4.0],
    [2.0, 2.3, 3.6, 4.2],
]
y_train = ["edible", "poisonous", "edible"]  # Example labels

# Initialize and train the KNN model
knn_model = MyKNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

@app.route("/")
def index():
    """Welcome page."""
    return "<h1>Welcome to the Mushroom Classifier App</h1>", 200

@app.route("/predict")
def predict():
    """Handle the prediction request."""
    # Parse the unseen instance values from the query string
    try:
        season = float(request.args.get("season", 0))
        stem_width = float(request.args.get("stem_width", 0))
        gill_color = float(request.args.get("gill_color", 0))
        cap_diameter = float(request.args.get("cap_diameter", 0))
    except ValueError:
        return jsonify({"error": "Invalid input. All parameters must be numeric."}), 400

    # Convert them into a list (normalized if needed)
    instance = [season, stem_width, gill_color, cap_diameter]

    # Make a prediction using the KNN model
    pred = knn_model.predict([instance])  # KNN expects a 2D array

    # Return the prediction in JSON format
    return jsonify({"prediction": pred[0]}), 200

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5001, debug=False)


# http://127.0.0.1:5001/predict?season=1.2&stem_width=0.5&gill_color=2.3&cap_diameter=3.5
# http://127.0.0.1:5001