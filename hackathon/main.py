import pickle
import numpy as np


model1 = pickle.load(open('WD_2.pkl', 'rb'))

temperature = 1
windspeed = 0
fuel_moisture_code = 0
duff_moisture_code = 0
initial_spread_index = 0
features = np.array([[temperature, windspeed, fuel_moisture_code, duff_moisture_code, initial_spread_index, 0, 0]])
# features = np.array([[0,0,0,0,0,0,0]])
# features = [temperature, windspeed, fuel_moisture_code, duff_moisture_code, initial_spread_index]
prediction1 = model1.predict_proba(features)
print(prediction1)
            # <h4 class=" fs-2 text-light text-bold" id="prediction" style="display: none; ">{{ pred }}</h4>
