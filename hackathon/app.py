from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model1 = pickle.load(open('WD_2.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        temperature = float(request.form['Temperature'])
        windspeed = float(request.form['Windspeed'])
        fuel_moisture_code = float(request.form['Fuel Moisture Code'])
        duff_moisture_code = float(request.form['Duff Moisture Code'])
        initial_spread_index = float(request.form['Initial Spread Index'])

        # Create a feature array
        features = np.array([[temperature, windspeed, fuel_moisture_code, duff_moisture_code, initial_spread_index, 0, 1]])

        # Make predictions using the loaded models
        prediction1 = model1.predict_proba(features)[0][1]
        print(prediction1)
        if prediction1 > 0.5:
            result1 = 'Your Forest is in Danger.\nProbability of fire occurring is {:.2f}'.format(prediction1)
        else:
            result1 = 'Your Forest is Safe.\nProbability of fire occurring is {:.2f}'.format(prediction1)

        return render_template('index.html', pred=result1)
    
    # For GET requests or initial page load
    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)

