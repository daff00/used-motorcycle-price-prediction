from flask import Flask, redirect, url_for, render_template, request
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("base.html")

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
    return render_template("base.html")

if __name__ == "__main__":
    app.run(debug=True)