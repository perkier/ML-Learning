import pandas as pd
import sys
import numpy as np


def find_path():
    # Return DATA Folder Path

    data_path = sys.path[0].split('London_Project')[0]
    data_path = f'{data_path}\\DATA\\Smart meters in London\\halfhourly_dataset\\'

    return data_path


def csv_func(name):

    csv_reader = open(f'{find_path() + name }.csv', 'rb')
    csv_read = pd.read_csv(csv_reader, encoding='latin1')
    csv_reader.close()

    # csv_read = csv_read.sample(frac=1).reset_index(drop=True)

    return csv_read


def normalize_energy_data(df,x):

    for elem in np.where(x == 999999999):

        for i in range(len(elem)):

            date_hour = df.iloc[elem[i]].loc['Hour']
            date_day = df.iloc[elem[i]].loc['Days']
            prev_day = date_day - 1
            block = df.iloc[elem[i]].loc['LCLid']

            looked_df = df.loc[df['Days'] == prev_day]
            # looked_df = looked_df.loc[df['Hour'] == date_hour]
            looked_df = looked_df.loc[df['LCLid'] == block]

            print(date_hour, date_day, block)

    return x


def see_all():
    # Alongate the view on DataFrames

    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)


def get_numdays(date):

    # List of previous days in each month
    days_premonths = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

    month = int(date.split('-')[0]) - 1

    days = days_premonths[month] + int(date.split('-')[1])

    return days


def transform_df(df, name):

    # Separate hours and minutes from the rest of the date
    new_dates = df['tstp'].str.split(" ", n=1, expand=True)

    # Separate year from months and days
    new_new_dates = new_dates[0].str.split("-", n=1, expand=True)

    df['Hour'] = new_dates.loc[:,1].str.split(".", n=1, expand=True).loc[:, 0]

    df['Year'] = new_new_dates.loc[:, 0]

    new_dates = new_new_dates[1]

    new_dates = new_dates.apply(lambda x: get_numdays(x))
    df['Days'] = new_dates

    # Drop the original date format
    df = df.drop(['tstp'], axis=1)

    df = df[df.loc[:,'energy(kWh/hh)'] != 'Null']

    energy_df = df.loc[:,'energy(kWh/hh)'].to_numpy()
    energy_df = energy_df.astype(float)

    df = df.drop(['energy(kWh/hh)'], axis=1)
    df['energy(kWh/hh)'] = energy_df

    df.to_csv(f"{find_path()}\\Transformed\\{name}.csv", index=False, header=True)

    return df


def looping_all_db(df):

    pass


def main():

    see_all()

    for i in range(112):

        name_block = f'block_{i}'

        block_0_df = csv_func(name_block)

        block_0_df = transform_df(block_0_df, name_block)

        print(i)


if __name__ == "__main__":

    main()
