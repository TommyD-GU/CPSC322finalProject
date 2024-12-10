import pickle
from flask import Flask, request, jsonify
from mysklearn.myclassifiers import MyKNeighborsClassifier  # Import your KNN classifier

app = Flask(__name__)

def load_model():
    """Load the trained KNN model from the pickle file."""
    with open("knn_model.p", "rb") as infile:  # Assuming 'knn_model.p' is where the model is pickled
        knn = pickle.load(infile)
    return knn

@app.route("/")
def index():
    """Welcome page."""
    return "<h1>Welcome to the Mushroom Classifier App</h1>", 200

@app.route("/predict")
def predict():
    """Handle the prediction request."""
    # Parse the unseen instance values from the query string
    season = request.args.get("season")
    stem_width = request.args.get("stem_width")
    gill_color = request.args.get("gill_color")
    cap_diameter = request.args.get("cap_diameter")

    # Convert them into a list (or normalize if needed)
    instance = [float(season), float(stem_width), float(gill_color), float(cap_diameter)]

    # Load the trained KNN model
    knn = load_model()

    # Make a prediction using the KNN model
    pred = knn.predict([instance])  # KNN expects a 2D array, hence wrapping the instance in another list

    # Return the prediction in JSON format
    return jsonify({"prediction": pred[0]}), 200

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5001, debug=False)
