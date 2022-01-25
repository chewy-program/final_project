import numpy as np
import os
import pandas as pd 
from flask import Flask, render_template, request, flash
# from flask_wtf import FlaskForm
from wtforms import IntegerField, BooleanField
import pickle #Initialize the flask App


app = Flask(__name__)
# secret_key = os.urandom(32)
# app.config["SECRET_KEY"] = "some string"
model = pickle.load(open('model.pkl', 'rb'))
# class predict_form(FlaskForm):
#     startingprice = IntegerField('Starting_Price')
#     owners = IntegerField('Owners')
#     age = IntegerField('age')
#     multiplayer = BooleanField("Multiplayer")
#     DLC = BooleanField("DLC")
# form = predict_form
#default page of our web-app
@app.route('/')
def home():
    return render_template("index.html")

#To use the predict button in our web-app
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    #For rendering results on HTML GUI
    int_features = [x for x in request.form.values()]
    final_features = np.array(int_features)
    df = pd.DataFrame([final_features])
    prediction = model.predict(df)
    # prediction = int(prediction.Label[0])
    value = round(prediction[0], 2)
    output = value/10 
    return render_template("index.html", prediction_text='predicted cost of the game is: ${:.2f}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)