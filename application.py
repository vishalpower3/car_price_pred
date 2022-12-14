from flask import Flask, render_template, request
import sklearn
import pandas as pd
import pickle
import numpy as np

car = pd.read_csv(r"Cleaned_data.csv")

with open("LinearRegressionModel.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    companies = sorted(car["company"].unique())
    car_models = sorted(car["name"].unique())
    year = sorted(car["year"].unique(), reverse=True)
    fuel_types = sorted(car["fuel_type"].unique())
    companies.insert(0, "Select Company")
    return render_template('index.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_types)

@app.route('/predict', methods= ['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))

    prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    return str(np.around(prediction[0],2))

if __name__ == "__main__":
    app.run(host= "0.0.0.0")