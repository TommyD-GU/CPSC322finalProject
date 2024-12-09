from flask import Flask
import test_web_app_things  # Import the MOMO file here

app = Flask(__name__)

@app.route('/')
def home():
    return "mo is an alt girl"  # Directly return the message from the home function

if __name__ == '__main__':
    app.run(debug=True)