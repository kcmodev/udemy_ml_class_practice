import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

import time


# def main():
# snp_data = pd.read_csv('data/snp_500_data.csv', parse_dates=["Date"])
snp_data = pd.read_csv('data/11_year_snp_500.csv', parse_dates=["Date"])

snp_data.sort_values(by=["Date"], inplace=True, ascending=True)
snp_data["dataPointYear"] = snp_data["Date"].dt.year
snp_data["dataPointMonth"] = snp_data["Date"].dt.month
snp_data["dataPointDay"] = snp_data["Date"].dt.day
snp_data["dataPointDayOfWeek"] = snp_data["Date"].dt.dayofweek
snp_data["dataPointDayOfYear"] = snp_data["Date"].dt.dayofyear
snp_data = snp_data.drop("Date", axis=1)

# split data into training, validation, and test sets
# print(f'Value Counts:\n {snp_data.dataPointYear.value_counts()}') # check for total values per year
snp_val = snp_data[snp_data.dataPointYear == 2015]
snp_train = snp_data[snp_data.dataPointYear != 2015]
print(f'Len of validation: {len(snp_val)}. Len of training: {len(snp_train)}')

x_train, y_train = snp_train.drop("Close", axis=1), snp_train.Close
x_valid, y_valid = snp_val.drop("Close", axis=1), snp_val.Close

# check both sets are same size and check y_train for correct type
# print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)
# print(y_train)

print('Dataset info:')
print(snp_data.info())

# check for cells with missing values
# print(f'\nis na.... \n{snp_data.isna().sum()}')

model = RandomForestRegressor(n_jobs=-1, random_state=27)
start_time = time.time()
model.fit(snp_data.drop("Close", axis=1), snp_data["Close"])
stop_time = time.time()
print(f'Total training time of {stop_time - start_time}s.')

model_score = model.score(snp_data.drop("Close", axis=1), snp_data["Close"])
print(f'Model score: {model_score}')

# fig, ax = plt.subplots()
# ax.plot(snp_data["Date"], snp_data["Close"])
# ax.set(title='S&P 500 price over time',
#        xlabel='Date',
#        ylabel='Price')
# plt.show()


def rmsle(y_test, y_pred):
    # Calculate ROOT mean squared log error
    return np.sqrt(mean_squared_log_error(y_test, y_pred))


def show_scores(model):
    train_preds = model.predict(x_train)
    # want to see slightly worse score on validation
    # if val is better than test then the model is probably over fitting
    valid_preds = model.predict(x_valid)
    scores = {"Training MAE: ": mean_absolute_error(y_train, train_preds),
              "Valid MAE: ": mean_absolute_error(y_valid, valid_preds),
              "Training RMSLE: ": rmsle(y_train, train_preds),
              "Validation RMSLE: ": rmsle(y_valid, valid_preds),
              "Training R^2: ": r2_score(y_train, train_preds),
              "Valid R^2: ": r2_score(y_valid, valid_preds)}

    return scores


if __name__ == '__main__':
    print('Running main...')
    # main()


# x = np.random.randn(1000)
#
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2,
#                                              ncols=2,
#                                              figsize=(10, 5))
#
# ax1.plot(x, x/2)
# ax2.scatter(np.random.random(10), np.random.random(10))
# plt.show()

# print(snp_data)
# print('*' * 50)
# print('\tdtypes...')
# print(snp_data.dtypes)
# print('*' * 50)
# print('\tinfo...')
# print(snp_data.info())
# print('*' * 50)
# print('\tdescribe...')
# print(snp_data.describe())
# print('*' * 50)
# print('\tclose mean...')
# print(snp_data['Close'].mean())
#
# # plt.plot(snp_data["Close"][:50])
# # plt.xlabel('Days since start')
# # plt.ylabel('Close stock price')
# # plt.show()
#
# print('*' * 50)
# print('\tnp array...')
# print(np.array(pd.read_csv('snp_500_data.csv')))

# x = [1, 2, 3, 4]
# y = [11, 22, 33, 44]
#
# fig, ax = plt.subplots()
# ax.plot(x, y)
# ax.set(title="Random Title",
#        xlabel="x axis label",
#        ylabel="y axis label")
# plt.show()
# fig.savefig('test_plot')
