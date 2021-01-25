import pandas as pd


def main():
    # axis 0 = row, axis 1 = column
    snp_data = pd.read_csv("snp_500_data.csv")
    print(snp_data)
    print('*' * 50)
    print('\tdtypes.....')
    print(snp_data.dtypes)
    print('*' * 50)
    print('\tinfo.....')
    print(snp_data.info())
    print('*' * 50)
    print('\tdescribe.....')
    print(snp_data.describe())
    print('*' * 50)
    print('\tclose mean.....')
    print(snp_data['Close'].mean())


if __name__ == '__main__':
    print('Running main...')
    main()

