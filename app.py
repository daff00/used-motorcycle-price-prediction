from flask import Flask, redirect, url_for, render_template, request
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

app = Flask(__name__)

le = LabelEncoder()
ss = StandardScaler()

@app.route('/')
def index():
    # Redirect to the home page
    return redirect(url_for('home'))

@app.route("/home")
def home():
    return render_template("base.html")

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, -1)
    loaded_model = pickle.load(open("nn_model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        json_str = json.dumps(to_predict_list)
        #to_predict_list = list(to_predict_list.values())
        # result = ValuePredictor(to_predict_list)      
    # return render_template("base.html", result = result)
    return json_str

if __name__ == "__main__":
    app.run(debug=True)