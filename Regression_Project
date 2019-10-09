import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn import preprocessing, neighbors, linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from sklearn import svm

from sklearn.linear_model import BayesianRidge

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


def take_time(last_time=None, print_me=None):

    func_name = traceback.extract_stack(None, 2)[0][2]

    if last_time == None:

        current_time = time.time()
        return current_time

    else:

        time_spent = time.time() - float(last_time)

        if print_me == None:
            pass

        else:
            print(f"--- {time_spent} seconds ---")

        funcs_time_line(func_name, time_spent)
        return time_spent


def funcs_time_line(func_name, time_spent):

    newpath = os.path.abspath(__file__).split('.')[0] + '.csv'

    if not os.path.exists(newpath):
        os.makedirs(newpath)

    csv_reader = open(newpath, 'rb')
    csv_read = pd.read_csv(csv_reader_1, encoding='latin1')
    csv_reader.close()

    try:
        csv_read.loc[func_name]

    except NameError:

        new_func_pd = pd.DataFrame({'Func_Name': func_name,
                                    'Time_Spent': time_spent,
                                    'Times_Ran': [1]})

        csv_read.append(new_func_pd, ignore_index=True)

    else:

        # csv_read.at[func_name, 'Func_Name'] = csv_read.loc[csv_read['Func_Name'] == func_name]
        pass


def csv_func(num, city):

    fatura_euro = f'{num}euro'

    csv_reader_1 = open(f'C:\\Users\\Diogo Sá\\Desktop\\perkier tech\\Energy\\Final_Calcs\\{city}\\Precision\\{fatura_euro}__panels_1.csv', 'rb')
    csv_read_1 = pd.read_csv(csv_reader_1, encoding='latin1')
    csv_reader_1.close()

    csv_reader_0 = open(f'C:\\Users\\Diogo Sá\\Desktop\\perkier tech\\Energy\\Final_Calcs\\{city}\\Precision\\{fatura_euro}__panels_0.csv', 'rb')
    csv_read_0 = pd.read_csv(csv_reader_0, encoding='latin1')
    csv_reader_0.close()

    csv_reader_m1 = open(f'C:\\Users\\Diogo Sá\\Desktop\\perkier tech\\Energy\\Final_Calcs\\{city}\\Precision\\{fatura_euro}__panels_m1.csv', 'rb')
    csv_read_m1 = pd.read_csv(csv_reader_m1, encoding='latin1')
    csv_reader_m1.close()

    return csv_read_1, csv_read_0, csv_read_m1


def plotting_hist(df):

    df.hist(bins=50, figsize=(20,15))
    plt.show()


def get_year(df):

    year_array = {}

    for i in range(len(df)):

        Module_name = df.iloc[i].loc['Module']
        splited_array = Module_name.split(' ')

        arrary_length = int(len(splited_array)) -1

        year_array[i] = int(splited_array[arrary_length])

    year_series = pd.Series(year_array)

    return year_series


def get_brand(df):

    brand_array = {}

    for i in range(len(df)):

        Module_name = df.iloc[i].loc['Module']
        splited_array = Module_name.split(' ')

        brand_array[i] = splited_array[1]

    brand_series = pd.Series(brand_array)

    return brand_series


def get_correlation(df):

    corr = pd.DataFrame(df.corr())

    return corr


def create_df(num, city):

    df_1, df_0, df_m1 = csv_func(num, city)

    df_0 = df_0.drop(columns='Sandia_Name')
    df_1 = df_1.drop(columns='Sandia_Name')
    df_m1 = df_m1.drop(columns='Sandia_Name')

    df_all = df_0.copy()
    df_all = df_all.append(df_1, ignore_index=True)
    df_all = df_all.append(df_m1, ignore_index=True)

    df_all['Total_Power'] = df_all['Recomended_Panels'] * df_all['Power']

    df_all['Year'] = get_year(df_all)
    df_all['Brand'] = get_brand(df_all)

    df_all = df_all.drop('Savings', axis=1)
    df_all = df_all.drop('Recom_Panels_Savings', axis=1)
    df_all = df_all.drop('Module', axis=1)

    # df_all = df_all.drop('Recomended_Panels', axis=1)
    df_all = df_all.drop('Total_Power', axis=1)

    # df_all = df_all.drop('Recomended_Panels', axis=1)
    # df_all = df_all.drop('Power', axis=1)

    return df_all


def split_train_test(data, test_ratio):

    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]


def create_encoder(df):

    encoder = LabelEncoder()
    df_cat = df["Brand"]
    df_cat_encoded = encoder.fit_transform(df_cat)

    encoder = OneHotEncoder(categories='auto')
    df_cat_1hot = encoder.fit_transform(df_cat_encoded.reshape(-1, 1))
    df_cat_array = df_cat_1hot.toarray()

    Canadian = {}
    LG = {}
    Panasonic = {}
    SunPower = {}
    Trina = {}

    for i in range(len(df)):

        Canadian[i] = df_cat_array[i][0]
        LG[i] = df_cat_array[i][1]
        Panasonic[i] = df_cat_array[i][2]
        SunPower[i] = df_cat_array[i][3]
        Trina[i] = df_cat_array[i][4]

    brands = pd.DataFrame({'Canadian': Canadian,
                           'LG': LG,
                           'Panasonic': Panasonic,
                           'SunPower': SunPower,
                           'Trina': Trina})

    return brands


def test_scores(clf, X_test, y_test, savings_train, savings_labels_train, savings_test, savings_labels_test):

    accuracy = clf.score(X_test, y_test) *100
    print(f'Accuracy Score: {accuracy}%')

    y_pred = clf.predict(X_test)

    predicted_df = pd.DataFrame({'Actual': y_test,
                                 'Predicted': y_pred})

    predicted_df['dif'] = (predicted_df['Actual'] - predicted_df['Predicted']) / predicted_df['Actual']

    diff_final = (1 - abs(predicted_df['dif'].mean())) * 100

    print(f'Percentage Difference (Predicted vs. Actual): {diff_final}%')

    scores = cross_val_score(clf, savings_train, savings_labels_train, cv=10)
    scores = scores.mean() * 100
    print(f'Cross Validation Score: {scores}%')

    scores = cross_val_score(clf, savings_test, savings_labels_test,
                             scoring="neg_mean_squared_error", cv=10)

    rmse_scores = np.sqrt(-scores)
    rmse_perc = (1 - (rmse_scores.mean() / predicted_df['Actual'].mean())) * 100
    print(f'Mean squared error: {rmse_perc}%')

    param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}, ]

    forest_reg = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                                       max_features=8, max_leaf_nodes=None, min_impurity_decrease=0.0,
                                       min_impurity_split=None, min_samples_leaf=1,
                                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                                       n_estimators=30, n_jobs=None, oob_score=False,
                                       random_state=None, verbose=0, warm_start=False)

    forest_reg.fit(savings_train, savings_labels_train)
    y_pred = forest_reg.predict(X_test)

    predicted_df = pd.DataFrame({'Actual': y_test,
                                 'Predicted': y_pred})

    predicted_df['dif'] = (predicted_df['Actual'] - predicted_df['Predicted']) / predicted_df['Actual']

    diff_final = (1 - abs(predicted_df['dif'].mean())) * 100

    print(f'RandomForestRegressor (Most Accurate): {diff_final}%')
    print('\n')

    # grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
    #                            scoring='neg_mean_squared_error')
    #
    # grid_search.fit(savings_train, savings_labels_train)
    #
    # print(grid_search.best_params_)
    # print(grid_search.best_estimator_)


def savings_prec_AI(df_all_mont_avr):

    df_all_mont_avr = df_all_mont_avr.drop('Sold_Grid_kWh', axis=1)
    df_all_mont_avr = df_all_mont_avr.drop('Bought_Grid_kWh', axis=1)
    df_all_mont_avr = df_all_mont_avr.drop('Yearly Prod. (W hr)', axis=1)

    param = 'Production_Precision'

    return df_all_mont_avr, param


def sold_grid_AI(df_all_mont_avr):

    df_all_mont_avr = df_all_mont_avr.drop('Production_Precision', axis=1)
    df_all_mont_avr = df_all_mont_avr.drop('Bought_Grid_kWh', axis=1)
    df_all_mont_avr = df_all_mont_avr.drop('Yearly Prod. (W hr)', axis=1)

    param = 'Sold_Grid_kWh'


    return df_all_mont_avr, param


def bought_grid_AI(df_all_mont_avr):

    df_all_mont_avr = df_all_mont_avr.drop('Sold_Grid_kWh', axis=1)
    df_all_mont_avr = df_all_mont_avr.drop('Production_Precision', axis=1)
    df_all_mont_avr = df_all_mont_avr.drop('Yearly Prod. (W hr)', axis=1)

    param = 'Bought_Grid_kWh'

    return df_all_mont_avr, param


def yearly_prod_AI(df_all_mont_avr):

    df_all_mont_avr = df_all_mont_avr.drop('Production_Precision', axis=1)
    df_all_mont_avr = df_all_mont_avr.drop('Bought_Grid_kWh', axis=1)
    df_all_mont_avr = df_all_mont_avr.drop('Sold_Grid_kWh', axis=1)

    param = 'Yearly Prod. (W hr)'


    return df_all_mont_avr, param


def splitting(df_all_mont_avr):

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # for train_index, test_index in split.split(df_all_mont_avr, df_all_mont_avr["Euros_Without_Panels"]):
    for train_index, test_index in split.split(df_all_mont_avr, df_all_mont_avr["Latitude"]):

        strat_train_set = df_all_mont_avr.loc[train_index]
        strat_test_set = df_all_mont_avr.loc[test_index]

    return strat_train_set, strat_test_set


def feature_scaling(df):

    brands = df.loc[:, 'Brand']
    df = df.drop(columns='Brand')

    df_np = df.to_numpy()

    mean_df = df.describe().loc['mean']
    mean_np = mean_df.to_numpy()

    max_df = df.describe().loc['max']
    max_np = max_df.to_numpy()

    min_df = df.describe().loc['min']
    min_np = min_df.to_numpy()

    max_np = max_np - min_np

    scaling_vector = (df_np - mean_np) / max_np

    df = pd.DataFrame(data= scaling_vector,
                      index = np.array([i for i in range(len(scaling_vector))]),
                      columns = list(df.columns.values))

    descaled_df = pd.DataFrame(data=[mean_np, max_np],
                              index=np.array([i for i in range(2)]),
                              columns=list(df.columns.values))

    df = df.join(brands)

    return df, descaled_df


def descaling(df, descaled_df, type):

    df = descaled_df.iloc[1].loc[type] * df + descaled_df.iloc[0].loc[type]

    return df


def learning_alg(df_all_mont_avr, ML_param):

    df_all_corr = get_correlation(df_all_mont_avr)

    df_all_sp = df_all_corr.loc[:,ML_param].dropna()

    df_all_corr = pd.DataFrame()

    df_all_corr[ML_param] = df_all_sp
    df_all_corr = df_all_corr.sort_values(by=ML_param)

    # print('\n'*3)
    # print(df_all_corr)
    # print('\n'*3)

    strat_train_set, strat_test_set = splitting(df_all_mont_avr)

    # for set in (strat_train_set, strat_test_set):
    #
    #     set.drop(["Euros_Without_Panels"], axis=1, inplace=True)

    # df_all_mont_avr.plot(kind="scatter", x=ML_param, y="Euros_Without_Panels", alpha=0.1)
    # plt.show()

    savings_train = strat_train_set.drop(ML_param, axis=1)
    savings_labels_train = strat_train_set[ML_param].copy()

    savings_test = strat_train_set.drop(ML_param, axis=1)
    savings_labels_test = strat_train_set[ML_param].copy()

    # print('\n'*3)
    #Cross Validation

    X = df_all_mont_avr.drop(ML_param, axis=1)                 #Features
    y = df_all_mont_avr[ML_param].copy()                       #Labels

    X_train = savings_train
    y_train = savings_labels_train
    X_test = savings_test
    y_test = savings_labels_test

    # X_train = np.array([X_train])
    # y_train = np.array([y_train])
    # X_test = np.array([X_test])
    # y_test = np.array([y_test])

    # clf = neighbors.KNeighborsClassifier(n_jobs=-1)
    # clf = linear_model.SGDRegressor(n_jobs = -1)
    # clf = linear_model.LinearRegression(n_jobs = -1)
    # clf = linear_model.LassoLars()
    # clf = linear_model.Ridge()

    # clf = linear_model.ElasticNet()

    # clf = RandomForestRegressor()
    # clf = DecisionTreeRegressor()
    # clf = svm.SVR()
    clf = GradientBoostingRegressor(loss='ls', alpha=0.95,
                                    n_estimators=250, max_depth=3,
                                    learning_rate=.1, min_samples_leaf=9,
                                    min_samples_split=9)

    # clf = BayesianRidge(compute_score=True)

    clf.fit(X_train, y_train)

    test_scores(clf, X_test, y_test, savings_train, savings_labels_train, savings_test, savings_labels_test)

    return clf


def city_loop(geo_city):

    lx_df = pd.DataFrame()

    for i in range(50, 80, 5):

        lx_df = lx_df.append(create_df(i, geo_city), ignore_index=True)

    return  lx_df




def regression_3d():

    pass



def lisbon_test(measures):

    city = 'Paris'

    measures_df = pd.DataFrame({'Module_Price': measures[0][0],
                                'Power': int(measures[0][1]),
                                'Recomended_Panels': int(measures[0][2]),
                                'Latitude': measures[0][3],
                                'Longitude': measures[0][4],
                                'kWh_Without_Panels': measures[0][5],
                                'Year': int(measures[0][6]),
                                'Canadian': measures[0][7],
                                'LG': measures[0][8],
                                'Panasonic': measures[0][9],
                                'SunPower': measures[0][10],
                                'Trina': measures[0][11]}, index=[0])

    lx_df = city_loop(city)

    lx_encoded = create_encoder(lx_df)
    lx_df = lx_df.join(lx_encoded)
    lx_df = lx_df.drop('Brand', axis=1)

    lx_chopped = lx_df.drop('Sold_Grid_kWh', axis=1)
    lx_chopped = lx_chopped.drop('Bought_Grid_kWh', axis=1)
    lx_chopped = lx_chopped.drop('Yearly Prod. (W hr)', axis=1)
    lx_chopped = lx_chopped.drop('Production_Precision', axis=1)

    diff = {}
    idx_lx = 0

    try:
        searched = lx_chopped.loc[lx_chopped['Module_Price'] == measures_df.iloc[0].loc['Module_Price']]
        searched = searched.loc[searched['Power'] == measures_df.iloc[0].loc['Power']]
        searched = searched.loc[searched['Latitude'] == measures_df.iloc[0].loc['Latitude']]
        searched = searched.loc[searched['Longitude'] == measures_df.iloc[0].loc['Longitude']]
        searched = searched.loc[searched['kWh_Without_Panels'] == measures_df.iloc[0].loc['kWh_Without_Panels']]
        searched = searched.loc[searched['Year'] == measures_df.iloc[0].loc['Year']]
        searched = searched.loc[searched['Canadian'] == measures_df.iloc[0].loc['Canadian']]
        searched = searched.loc[searched['LG'] == measures_df.iloc[0].loc['LG']]
        searched = searched.loc[searched['Panasonic'] == measures_df.iloc[0].loc['LG']]
        searched = searched.loc[searched['Trina'] == measures_df.iloc[0].loc['Trina']]
        searched = searched.loc[searched['SunPower'] == measures_df.iloc[0].loc['SunPower']]
        searched = searched.loc[searched['Recomended_Panels'] == measures_df.iloc[0].loc['Recomended_Panels']].index.values.astype(int)[0]

        idx_lx = searched

    except:

        for i in range(len(lx_chopped)):

            colm = {}
            tot = 0
            j = 0

            for col in lx_chopped.columns:

                colm[j] = col
                j += 1

                diff[j] = lx_chopped.iloc[i].loc[col] - measures_df.iloc[0].loc[col]

                tot += diff[j]

            if abs(tot) < 0.1:

                idx_lx = i
                break

    lx_savings = lx_df.iloc[idx_lx].loc['Production_Precision']
    lx_bought = lx_df.iloc[idx_lx].loc['Bought_Grid_kWh']
    lx_sold = lx_df.iloc[idx_lx].loc['Sold_Grid_kWh']
    lx_year_prod = lx_df.iloc[idx_lx].loc['Yearly Prod. (W hr)']

    return lx_savings, lx_bought, lx_sold, lx_year_prod


def lx_comparisson(example_measures, pred_sav_prec, pred_sold_grid, pred_bougth_grid):

    lx_savings, lx_bought, lx_sold, lx_year_prod = lisbon_test(example_measures)

    print('\n')
    print(lx_savings, lx_bought, lx_sold)
    print(pred_sav_prec[0], pred_bougth_grid[0], pred_sold_grid[0])


    error_savings = 100 * abs((lx_savings - pred_sav_prec[0]) / lx_savings)
    error_sold = 100 * abs((lx_sold - pred_sold_grid[0]) / lx_sold)
    error_bought = 100 * abs((lx_bought - pred_bougth_grid[0]) / lx_bought)

    # print(f'Production_Error of {error_savings}%, i.e. a precision of {100-error_savings}%')
    # print(f'Sold_kWh_Error of {error_sold}%, i.e. a precision of {100-error_sold}%')
    # print(f'Bought_kWh_Error of {error_bought}%, i.e. a precision of {100-error_bought}%')
    # print('\n')

    return error_savings, error_sold, error_bought


def example(df):

    example_measures = np.array([[df.loc['Module_Price'],
                                  df.loc['Power'],
                                  df.loc['Recomended_Panels'],
                                  df.loc['Latitude'],
                                  df.loc['Longitude'],
                                  df.loc['kWh_Without_Panels'],
                                  df.loc['Year'],
                                  df.loc['Canadian'],
                                  df.loc['LG'],
                                  df.loc['Panasonic'],
                                  df.loc['SunPower'],
                                  df.loc['Trina']]])

    example_measures = example_measures.reshape(len(example_measures),-1)

    return example_measures


def encode_all(df):

    # mean_df = df.describe().loc['mean']
    mean_df = df.describe().loc['max']
    mean_np = mean_df.to_numpy()

    new_df = df / mean_np

    return new_df, mean_df


def decoded(pred_sav_prec, pred_sold_grid, pred_bougth_grid, mean_df):

    pred_sav_prec = mean_df.loc['Production_Precision'] * pred_sav_prec
    pred_sold_grid = mean_df.loc['Sold_Grid_kWh']* pred_sold_grid
    pred_bougth_grid = mean_df.loc['Bought_Grid_kWh'] * pred_bougth_grid

    return pred_sav_prec, pred_sold_grid, pred_bougth_grid


def scaling_example(example, descaled_df, target):

    descaled_df = descaled_df.drop(columns=['Yearly Prod. (W hr)', 'Production_Precision', 'Sold_Grid_kWh', 'Bought_Grid_kWh'])

    ex_1 = example[0][(len(example[0])-5):]
    ex_2 = example[0][:-5]

    df_np = descaled_df.to_numpy()

    sv = (ex_2 - df_np[0]) / df_np[1]
    sv = np.concatenate((sv, ex_1), axis=None)

    return sv



# Scikit-Learn provides the Pipeline class to help with sequences of transformations

# num_pipeline = Pipeline([
#     ('imputer', Imputer(strategy="median")),
#     ('attribs_adder', CombinedAttributesAdder()),
#     ('std_scaler', StandardScaler()),])
#
# savings_num_tr = num_pipeline.fit_transform(housing_num)


def main():

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    df_all_mont_avr = pd.DataFrame()

    geo_city = {}
    geo_city = ['Lisboa', 'Berlin', 'Rome', 'Edinburgh']


    for j in range(len(geo_city)):

        for i in range(50, 105, 5):

            df_all_mont_avr = df_all_mont_avr.append(create_df(i, geo_city[j]), ignore_index=True)


    df_all_mont_avr, descaled_df = feature_scaling(df_all_mont_avr)

    brands_encoded = create_encoder(df_all_mont_avr)
    df_all_mont_avr = df_all_mont_avr.join(brands_encoded)
    df_all_mont_avr = df_all_mont_avr.drop('Brand', axis=1)

    # df_all_mont_avr, mean_df = encode_all(df_all_mont_avr)


    ML_param = {}
    df_all_changed = {}

    df_all_changed[0], ML_param[0] = savings_prec_AI(df_all_mont_avr)
    df_all_changed[1], ML_param[1] = sold_grid_AI(df_all_mont_avr)
    df_all_changed[2], ML_param[2] = bought_grid_AI(df_all_mont_avr)
    df_all_changed[3], ML_param[3] = yearly_prod_AI(df_all_mont_avr)

    print('\n')
    print(df_all_changed[0].head())
    print(df_all_changed[0].tail())
    print('\n')
    print(df_all_changed[1].head())
    print(df_all_changed[1].tail())
    print('\n')
    print(df_all_changed[2].head())
    print(df_all_changed[2].tail())
    print('\n')
    print(df_all_changed[3].head())
    print(df_all_changed[3].tail())
    print('\n')

    # plotting_hist(df_all_mont_avr)

    clf_sav_prec = learning_alg(df_all_changed[0], ML_param[0])
    clf_sold_grid = learning_alg(df_all_changed[1], ML_param[1])
    clf_bougth_grid = learning_alg(df_all_changed[2], ML_param[2])
    clf_year_prod = learning_alg(df_all_changed[3], ML_param[3])

    Prod_error_array = {}
    Sold_error_array = {}
    Bought_error_array = {}
    i_array = {}
    brand_array = {}

    start_time = time.time()

    error_df_1 = pd.DataFrame()

    j = 0

    Paris_df = pd.DataFrame()

    for i in range(50, 105, 5):

        Paris_df = Paris_df.append(create_df(i, 'Paris'), ignore_index=True)

    Paris_encoded = create_encoder(Paris_df)
    Paris_df = Paris_df.join(Paris_encoded)
    Paris_df = Paris_df.drop('Brand', axis=1)
    Paris_df = Paris_df.drop('Yearly Prod. (W hr)', axis=1)
    Paris_df = Paris_df.drop('Production_Precision', axis=1)
    Paris_df = Paris_df.drop('Sold_Grid_kWh', axis=1)
    Paris_df = Paris_df.drop('Bought_Grid_kWh', axis=1)

    length = len(Paris_df)

    # for i in range(1, length, 32):
    # for i in range(1, int(length/2), 16):
    for i in range(int(length / 2), length, 16):

        target = Paris_df.iloc[i]

        if target.loc['Canadian'] == 1:
            brand = 'Canadian'

        if target.loc['LG'] == 1:
            brand = 'LG'

        if target.loc['SunPower'] == 1:
            brand = 'SunPower'

        if target.loc['Trina'] == 1:
            brand = 'Trina'

        example_measures = example(target)

        example_scaled = [scaling_example(example_measures, descaled_df, target)]

        pred_sav_prec = clf_sav_prec.predict(example_scaled)
        pred_sold_grid = clf_sold_grid.predict(example_scaled)
        pred_bougth_grid = clf_bougth_grid.predict(example_scaled)

        pred_sav_prec = descaling(pred_sav_prec, descaled_df, 'Production_Precision')
        pred_sold_grid = descaling(pred_sold_grid, descaled_df, 'Sold_Grid_kWh')
        pred_bougth_grid = descaling(pred_bougth_grid, descaled_df, 'Bought_Grid_kWh')

        percentage_done = i / length * 100
        print(f'{percentage_done}%')
        # print(pred_sav_prec[0])
        # print(savings)

        strt_time = time.time()

        Prod_error_array[j], Sold_error_array[j], Bought_error_array[j] = lx_comparisson(example_measures, pred_sav_prec, pred_sold_grid, pred_bougth_grid)

        print("--- %s seconds --- lx_test" % (time.time() - strt_time))

        i_array[j] = i
        brand_array[j] = brand

        error_df_1 = error_df_1.append(target, ignore_index=True)

        j += 1

    error_df = pd.DataFrame({'Prod_error': Prod_error_array,
                             'Sold_error': Sold_error_array,
                             'Bought_error': Bought_error_array,
                             'Brand': brand_array,
                             'i': i_array})

    error_df_1 = error_df_1.join(error_df)

    error_df_1 = error_df_1.drop('Canadian', axis=1)
    error_df_1 = error_df_1.drop('LG', axis=1)
    error_df_1 = error_df_1.drop('Panasonic', axis=1)
    error_df_1 = error_df_1.drop('SunPower', axis=1)
    error_df_1 = error_df_1.drop('Trina', axis=1)

    print('\n')

    # print(sum(Prod_error_array) / len(Prod_error_array))
    # print(sum(Sold_error_array) / len(Sold_error_array))
    # print(sum(Bought_error_array) / len(Bought_error_array))

    print(error_df_1)

    print('\n')

    error_desc = error_df.describe()
    print(error_desc.loc['mean'])

    print("--- %s seconds ---" % (time.time() - start_time))

    quit()


if __name__ == "__main__":

    main()
