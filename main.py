import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    # axis 0 = row, axis 1 = column
    snp_data = pd.read_csv('snp_500_data.csv')
    print(snp_data)
    print('*' * 50)
    print('\tdtypes...')
    print(snp_data.dtypes)
    print('*' * 50)
    print('\tinfo...')
    print(snp_data.info())
    print('*' * 50)
    print('\tdescribe...')
    print(snp_data.describe())
    print('*' * 50)
    print('\tclose mean...')
    print(snp_data['Close'].mean())

    plt.plot(snp_data["Close"][:50])
    plt.xlabel('Days since start')
    plt.ylabel('Close stock price')
    plt.show()

    print('*' * 50)
    print('\tnp array...')
    print(np.array(pd.read_csv('snp_500_data.csv')))


if __name__ == '__main__':
    print('Running main...')
    main()

