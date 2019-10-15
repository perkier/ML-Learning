import pandas as pd
import sys
import numpy as np


def find_path():
    # Return DATA Folder Path

    data_path = sys.path[0].split('London_Project')[0]
    data_path = f'{data_path}\\DATA\\Smart meters in London\\'

    return data_path


def csv_func(name):

    name = f'{name}'

    csv_reader = open(f'{find_path()}{name}.csv', 'rb')
    csv_read = pd.read_csv(csv_reader, encoding='latin1')
    csv_reader.close()

    # csv_read = csv_read.sample(frac=1).reset_index(drop=True)

    return csv_read


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


def transform_time(df):

    try:

        # Separate hours and minutes from the rest of the date
        new_dates = df['time'].str.split(" ", n=1, expand=True)
        df['Hour'] = new_dates.loc[:, 1]

    except:
        pass

    # Separate year from months and days
    new_new_dates = new_dates[0].str.split("-", n=1, expand=True)
    df['Year'] = new_new_dates.loc[:, 0]

    new_dates = new_new_dates[1]

    new_dates = new_dates.apply(lambda x: get_numdays(x))
    df['Days'] = new_dates

    # Drop the original date format
    df = df.drop(['time'], axis=1)

    return df


def csv_data_transformed(i):

    name = f'block_{i}'

    csv_reader = open(f'C:\\Users\\diogo\\OneDrive\\Área de Trabalho\\perkier tech\\Energy\\DATA\\Smart meters in London\\halfhourly_dataset\\Transformed\\{name}.csv', 'rb')
    csv_read = pd.read_csv(csv_reader, encoding='latin1')
    csv_reader.close()

    # csv_read = csv_read.sample(frac=1).reset_index(drop=True)

    return csv_read


def transform_half_hourly(weather_df):

    visibility = {}
    windBearing = {}
    temperature = {}
    dewPoint = {}
    pressure = {}
    apparentTemperature = {}
    windSpeed = {}
    precipType = {}
    icon = {}
    humidity = {}
    summary = {}
    Hour = {}
    Year = {}
    Days = {}

    columns = [visibility,windBearing,temperature,dewPoint,pressure,apparentTemperature,windSpeed,precipType,icon,humidity,summary,Hour,Year,Days]
    names = ['visibility', 'windBearing', 'temperature', 'dewPoint', 'pressure', 'apparentTemperature', 'windSpeed', 'precipType', 'icon', 'humidity', 'summary', 'Hour', 'Year', 'Days']

    median = ['visibility', 'windBearing', 'temperature', 'dewPoint', 'pressure', 'apparentTemperature', 'windSpeed', 'humidity']
    strings = ['precipType', 'icon', 'summary', 'Year', 'Days']
    Hours = ['Hour']

    k = 0

    for i in range(len(weather_df)-1):

        jj = 0

        for j in columns:

            j[k] = weather_df.iloc[i].loc[names[jj]]
            # print(f'{j[k]}')
            # print(type(j[k]))

            jj += 1

        k += 1
        jj = 0

        for j in columns:

            # Selects parameters that are in the median array
            if any(x == names[jj] for x in median):

                # Does the median value between two hours
                j[k] = (weather_df.iloc[i].loc[names[jj]] + weather_df.iloc[i+1].loc[names[jj]])/2

            # Selects parameters that are in the string array
            if any(x == names[jj] for x in strings):

                # Keeps the values of the last data point
                j[k] = weather_df.iloc[i].loc[names[jj]]

            # Selects parameters that are in the hours array
            if any(x == names[jj] for x in Hours):

                # Creates the half past the hour
                hour_splitted = weather_df.iloc[i].loc[names[jj]].split(':')
                j[k] = f'{hour_splitted[0]}:30:{hour_splitted[2]}'

            jj += 1

        k += 1

    new_weater_df = pd.DataFrame({'visibility': visibility,
                                  'windBearing': windBearing,
                                  'temperature': temperature,
                                  'dewPoint': dewPoint,
                                  'pressure': pressure,
                                  'apparentTemperature': apparentTemperature,
                                  'windSpeed': windSpeed,
                                  'precipType': precipType,
                                  'icon': icon,
                                  'humidity': humidity,
                                  'summary': summary,
                                  'Hour': Hour,
                                  'Year': Year,
                                  'Days': Days})

    return new_weater_df



def combine_weather_data(weather_df, data_df):

    new_df = pd.merge(data_df, weather_df,  how='left', left_on=['Hour','Year', 'Days'], right_on = ['Hour','Year', 'Days'])

    # Hour = data_df.iloc[1000].loc['Hour']
    # Year = data_df.iloc[1000].loc['Year']
    # Days = data_df.iloc[1000].loc['Days']
    #
    # target_weather = weather_df.loc[weather_df['Year'] == Year]
    # target_weather = target_weather.loc[target_weather['Days'] == Days]
    # target_weather = target_weather.loc[target_weather['Hour'] == Hour]
    #
    # target_data = data_df.loc[data_df['Year'] == Year]
    # target_data = target_data.loc[target_data['Days'] == Days]
    # target_data = target_data.loc[target_data['Hour'] == Hour]

    return new_df


def combine_hh_info(combined_df, households_info):

    new_df = pd.merge(combined_df, households_info, how='left', left_on=['LCLid'],
                      right_on=['LCLid'])

    new_df = new_df.drop(['stdorToU', 'Acorn_grouped', 'file', 'LCLid'], axis=1)

    return new_df


def combine_holidays(weather_hourly, uk_holidays):

    Holiday_0_1 = np.ones(len(uk_holidays))

    uk_holidays['Holiday'] = Holiday_0_1

    uk_holidays = uk_holidays.drop(['Type'], axis=1)

    uk_holidays['Year'] = uk_holidays['Year'].apply(lambda x: int(x))
    uk_holidays['Days'] = uk_holidays['Days'].apply(lambda x: int(x))

    new_df = pd.merge(weather_hourly, uk_holidays, how='left', left_on=['Year', 'Days'],
                      right_on=['Year', 'Days'])

    new_df = new_df.fillna(0)

    return new_df


def retune_weather_df(weather_df):

    # Get index values of values which have NaN - In Pressure
    NaN_values_df = weather_df[weather_df.isna().any(axis=1)].index

    # Get Biggest correlation to Pressure values - windspeed
    # print(weather_df.corr())

    for i in NaN_values_df:

        windSpeed = weather_df.iloc[i].loc['windSpeed']
        target_weather = weather_df.loc[weather_df['windSpeed'] == windSpeed]

        pressure = target_weather.describe().loc[:,'pressure'].loc['mean']
        weather_df.at[i, 'pressure'] = pressure

    return weather_df


def save_combined(to_save_df,i):

    to_save_df.to_csv(f"{find_path()}\\Combined\\block_{i}.csv", index=False, header=True)

    print(f'block_{i} Saved')



def main():

    see_all()

    weather_hourly = csv_func('weather_hourly_darksky')
    weather_hourly = transform_time(weather_hourly)
    weather_hourly = retune_weather_df(weather_hourly)
    weather_hourly = transform_half_hourly(weather_hourly)
    weather_hourly['Year'] = weather_hourly['Year'].apply(lambda x: int(x))

    uk_holidays = csv_func('uk_bank_holidays')
    uk_holidays = uk_holidays.rename(columns={"Bank holidays": "time"})
    uk_holidays = transform_time(uk_holidays)

    weather_hourly = combine_holidays(weather_hourly, uk_holidays)

    for i in range(112):

        transformed_data = csv_data_transformed(i)

        combined_df = combine_weather_data(weather_hourly, transformed_data)

        households_info = csv_func('informations_households')
        households_info = households_info.loc[households_info['file'] == f'block_{i}']

        combined_df = combine_hh_info(combined_df, households_info)

        save_combined(combined_df, i)

    # print(combined_df.corr().loc[:,'energy(kWh/hh)'])

    # acorn_details = csv_func('acorn_details')
    # print(acorn_details)

    print('\n'*5)
    print('Done')


if __name__ == '__main__':

    main()
