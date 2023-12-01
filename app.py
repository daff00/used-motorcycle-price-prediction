from flask import Flask, redirect, url_for, render_template, request
import numpy as np
import pickle
import json

app = Flask(__name__)

# Import scaler
X_scaler = pickle.load(open('X_scaler.pkl', 'rb'))
y_scaler = pickle.load(open('y_scaler.pkl', 'rb'))

@app.route('/')
def index():
    # Redirect to the home page
    return redirect(url_for('home'))

# Home
@app.route("/home")
def home():
    return render_template("base.html")

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1,1)

    # Scale user input
    to_predict_scaled = X_scaler.fit_transform(to_predict)

    # Load model
    loaded_model = pickle.load(open("nn_model.pkl", "rb"))
    
    # Reshape to original shape
    backshape = to_predict_scaled.reshape(1,-1)

    # Predict
    result = loaded_model.predict(backshape)

    # Inverse transform the result
    result = y_scaler.inverse_transform(result)

    # Convert the result to a JSON
    result = result.tolist()
    result_json = json.dumps(result)
    return result_json

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        # User input to list
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())

        # Convert list dtype to float
        to_predict_list = [float(i) for i in to_predict_list]

        # predict
        result = ValuePredictor(to_predict_list)
    return render_template("predict.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)