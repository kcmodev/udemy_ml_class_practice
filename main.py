import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Classification ML libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import time

selected_city = 'Kansas City'

# Pull attributes from selected city only
city_attributes = pd.read_csv('weather_data/city_attributes.csv')
city_attributes = city_attributes.groupby(['City']).get_group(selected_city)

try:
    # Read previously generated dataframe for selected city
    city_df = pd.read_csv(f'saved_dataframes/{selected_city}_weather_data')
    print(f'File for {selected_city} already generated.')
    print(city_df)

except FileNotFoundError:
    print(f'File for {selected_city} not found. Generating now...')

    # Pull remaining climate data for the selected city
    city_humidity = pd.read_csv('weather_data/humidity.csv', parse_dates=["datetime"])
    city_pressure = pd.read_csv('weather_data/pressure.csv', parse_dates=["datetime"])
    city_temp = pd.read_csv('weather_data/temperature.csv', parse_dates=["datetime"])
    city_weather_desc = pd.read_csv('weather_data/weather_description.csv', parse_dates=["datetime"])
    city_wind_speed = pd.read_csv('weather_data/wind_speed.csv', parse_dates=["datetime"])
    city_wind_dir = pd.read_csv('weather_data/wind_direction.csv', parse_dates=["datetime"])

    # Combine all elements of each DataFrame into one to be used as the training set.
    # Drops any NaN elements as they will not be needed.
    city_df = pd.DataFrame({'Humidity': city_humidity[selected_city],
                            'Pressure': city_pressure[selected_city],
                            'Temperature': city_temp[selected_city],
                            'Wind Speed': city_wind_speed[selected_city],
                            'Wind Direction': city_wind_dir[selected_city],
                            'Weather Description': city_weather_desc[selected_city]}).dropna()

    city_df.to_csv(f'saved_dataframes/{selected_city}_weather_data', index=False)

# Drop the weather description since that is the variable to be predicted
x = city_df.drop('Weather Description', axis=1)

# Retrieve only the weather description to test against the model's predictions.
y = city_df['Weather Description']

# Split x and y DataFrames into testing and training sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


clf = RandomForestClassifier(random_state=27)

print('Starting...')
start_time = time.time()
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
stop_time = time.time()
print(f'Score: {score}. Time to complete: {stop_time - start_time}')

data_columns = np.array(['Humidity', 'Pressure', 'Temperature', 'Wind Speed', 'Wind Direction'])
user_selections = np.array([70, 1001, 12, 10, 94]).reshape(1, -1)
print(clf.predict(user_selections[:]))


# def rmsle(y_test, y_pred):
#     # Calculate ROOT mean squared log error
#     return np.sqrt(mean_squared_log_error(y_test, y_pred))


# def show_scores(model):
#     train_preds = model.predict(x_train)
#     # want to see slightly worse score on validation
#     # if val is better than test then the model is probably over fitting
#     valid_preds = model.predict(x_valid)
#     scores = {"Training MAE: ": mean_absolute_error(y_train, train_preds),
#               "Valid MAE: ": mean_absolute_error(y_valid, valid_preds),
#               "Training RMSLE: ": rmsle(y_train, train_preds),
#               "Validation RMSLE: ": rmsle(y_valid, valid_preds),
#               "Training R^2: ": r2_score(y_train, train_preds),
#               "Valid R^2: ": r2_score(y_valid, valid_preds)}
#
#     for x, y in scores.items():
#         print(f'{x} \t\t.......{y}')

