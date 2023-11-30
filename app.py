from flask import Flask, redirect, url_for, render_template, request
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

app = Flask(__name__)

# Scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def index():
    # Redirect to the home page
    return redirect(url_for('home'))

@app.route("/home")
def home():
    return render_template("base.html")

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list)
    to_predict_scaled = scaler.transform(to_predict)
    loaded_model = pickle.load(open("nn_model.pkl", "rb"))

    result = loaded_model.predict(to_predict_scaled)

    # Inverse transform the result
    result = scaler.inverse_transform(result)

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
        #json_str = json.dumps(to_predict_list)
        #json_str = json.dumps(to_predict_list)
        #to_predict_list = list(to_predict_list.values())
        result = ValuePredictor(to_predict_list)      
    # return render_template("base.html", result = result)
    return render_template("predict.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)