from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from psx import stocks, tickers
import datetime

app = Flask(__name__, template_folder='/app')

# Load the dataset
tickers = tickers()
file_path = "result.csv"

if os.path.exists(file_path):
    os.remove(file_path)
    print(f"{file_path} has been deleted.")
else:
    print(f"{file_path} does not exist.")





data = stocks(["ZELP", "ZAHID"], start=datetime.date(2018, 1, 1), end=datetime.date.today())
data.to_csv(r'result.csv')
data = pd.read_csv('result.csv')
# Split the data into training and testing sets (80% training, 20% testing)
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Define the input features (Open, High, Low, and Volume)
X_train = train_data[['Open', 'High', 'Low', 'Volume']]
X_test = test_data[['Open', 'High', 'Low', 'Volume']]

# Define the target variable (Close)
y_train = train_data['Close']
y_test = test_data['Close']

# Create polynomial features of degree 2
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train the polynomial regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Print the model's mean squared error on the test data
y_pred = model.predict(X_test_poly)
mse = mean_squared_error(y_test, y_pred)

# Save the trained model
joblib.dump(model, 'model.pkl')

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def chart():
    # Extract data for charting
    chart_data = data[['Date', 'High', 'Low']].values.tolist()

    # Make predictions on test data and compute evaluation metrics
    y_pred = model.predict(X_test_poly)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Render the template with the chart data and evaluation metrics
    return render_template('index.html', chart_data=chart_data, r2=r2, mse=mse, mae=mae)

@app.route('/input')
def myinput():
    # Render the input template
    return render_template('Input.html')

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Extract the input features from the form
        Open = float(request.form['Open'])
        Max = float(request.form['Max'])
        Min = float(request.form['Min'])
        Volume = float(request.form['Volume'])

        # Create a DataFrame with the input features and predict the output
        d = {'Open': [Open], 'High': [Max], 'Low': [Min], 'Volume': [Volume]}
        d = pd.DataFrame(d)
        result = model.predict(poly.fit_transform(d))

        # Render the prediction template with the result
        return render_template('prediction.html', prediction=result[0], output="")
    except:
        # Render the prediction template with an error message if the input features are invalid
        return render_template('prediction.html', prediction=0, output="Enter input features in the previous page")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)
