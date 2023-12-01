from flask import Flask, redirect, url_for, render_template, request
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import pickle
import json

app = Flask(__name__)

X_scaler_250more = pickle.load(open('X_scaler250more.pkl', 'rb'))
y_scaler_250more = pickle.load(open('y_scaler250more.pkl', 'rb'))

# Scaler250less
X_scaler_250less = pickle.load(open('X_scaler250less.pkl', 'rb'))
y_scaler_250less = pickle.load(open('y_scaler250less.pkl', 'rb'))

@app.route('/')
def index():
    # Redirect to the home page
    return redirect(url_for('home'))

@app.route("/home")
def home():
    return render_template("base.html")

def ValuePredictor_250less(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1,1)
    to_predict_scaled = X_scaler_250less.fit_transform(to_predict)
    loaded_model = pickle.load(open("nn_model_250less.pkl", "rb"))
    
    backshape = to_predict_scaled.reshape(1,-1)

    result = loaded_model.predict(backshape)

    # Inverse transform the result
    result = y_scaler_250less.inverse_transform(result)

    # Convert the result to a Python list
    result = result.tolist()

    result_json = json.dumps(result)
    return result_json

def ValuePredictor_250more(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(-1,1)
    to_predict_scaled = X_scaler.fit_transform(to_predict)
    loaded_model = pickle.load(open("nn_model250more.pkl", "rb"))
    
    backshape = to_predict_scaled.reshape(1,-1)

    result = loaded_model.predict(backshape)

    # Inverse transform the result
    result = y_scaler.inverse_transform(result)

    # Convert the result to a Python list
    result = result.tolist()

    result_json = json.dumps(result)
    return result_json

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = [float(i) for i in to_predict_list]
        if to_predict_list[2] < 250:
            result = ValuePredictor_250less(to_predict_list)
        else:
            result = ValuePredictor_250more(to_predict_list)
              
    return render_template("predict.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)