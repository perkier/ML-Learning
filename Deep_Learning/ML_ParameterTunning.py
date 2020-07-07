import pandas as pd
import numpy as np
from datetime import date, datetime
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.model_selection import ShuffleSplit, train_test_split, cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import RANSACRegressor, HuberRegressor, LinearRegression, ElasticNet, ElasticNetCV, Lars, Lasso, LassoLars, LassoLarsIC, OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, Ridge, SGDRegressor

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation, Conv2D, LSTM, Dropout
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras.backend as K
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.callbacks import EarlyStopping, History

def csv_func(name = 'augmented_results'):

    csv_reader = open(f'D:\\Diogo\\Simulations_Reports_1stOrd\\{name}.csv', 'rb')
    csv_read = pd.read_csv(csv_reader, encoding='latin1', delimiter=',', dtype=np.float64)
    csv_reader.close()

    return csv_read


def csv_TST(name = 'Results_TST'):

    csv_reader = open(f'D:\\Diogo\\Simulations_Reports_TST\\{name}.csv', 'rb')
    csv_read = pd.read_csv(csv_reader, encoding='latin1', delimiter=',', dtype=np.float64)
    csv_reader.close()

    return csv_read



def merge_dfs(ejector_df_Te10, ejector_df_Te15):

    ejector_df_Te10['T_e'] = pd.Series(np.full(len(ejector_df_Te10), float(10)))
    ejector_df_Te15['T_e'] = pd.Series(np.full(len(ejector_df_Te15), float(15)))

    ejector_concated = pd.concat([ejector_df_Te10, ejector_df_Te15]).sort_values(by=['T_e','T_g', 'spindle_pos'], ascending=True).reset_index(drop=True)

    return ejector_concated



def create_zero_values(df):

    # print(df)

    unique_tg = df.loc[:, 'T_g'].unique()
    unique_tc = df.loc[:, 'T_c'].unique()
    unique_mm = df.loc[:, 'spindle_pos'].unique()

    i = 0

    for tg in unique_tg:
        for mm in unique_mm:

            unique_df = df.loc[df['T_g'] == tg].loc[df['spindle_pos'] == mm]

            print(unique_df)
            print('\n')

    quit()



def see_all():
    # Alongate the view on DataFrames

    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000000)


def splitting(df, variable):

    split = ShuffleSplit(n_splits=1, test_size= 0.15, random_state=42)

    for train_index, test_index in split.split(df, df[variable]):

        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]

    return strat_train_set, strat_test_set


def define_trainig_set(df, variable="T_g"):

    strat_train_set, strat_test_set = splitting(df, variable)

    savings_train = strat_train_set.drop(["prim_mdot", "sec_mdot"], axis=1)
    savings_labels_train = strat_train_set[["prim_mdot", "sec_mdot"]].copy()

    savings_test = strat_test_set.drop(["prim_mdot", "sec_mdot"], axis=1)
    savings_labels_test = strat_test_set[["prim_mdot", "sec_mdot"]].copy()

    X_train = savings_train
    y_train = savings_labels_train
    X_test = savings_test
    y_test = savings_labels_test

    return X_train, y_train, X_test, y_test


def shape_critical_set(df):

    index_1 = df.loc[df['Critical'] == 1].index.values
    index_0 = df.loc[df['Critical'] == 0].index.values

    test_set = df.iloc[index_1].reset_index(drop=True)
    test_set = test_set.drop(['outlet', 'Critical'], axis=1)

    savings_test = test_set.drop('ER', axis=1)
    savings_labels_test = test_set['ER'].copy()

    X_test = savings_test
    y_test = savings_labels_test

    other_set = df.iloc[index_0].reset_index(drop=True)
    other_set = other_set.drop(['outlet', 'Critical'], axis=1)

    return other_set, X_test, y_test


def feature_scaling(df):

    Crit_df = df.loc[:,"Crit"]
    df = df.drop(['Crit'], axis=1)

    df_np = df.to_numpy()

    mean_df = df.describe().loc['mean']
    mean_np = mean_df.to_numpy()

    max_df = df.describe().loc['max']
    max_np = max_df.to_numpy()

    min_df = df.describe().loc['min']
    min_np = min_df.to_numpy()

    max_np = max_np - min_np

    std_df = df.describe().loc['std']
    std_np = std_df.to_numpy()

    # Standardization (or Z-score normalization)
    scaling_vector = (df_np - mean_np) / std_np

    scaled_array = {'instructions': '(df_np - mean_np) / max_np', 'mean_np': mean_np, 'std_np': std_np}

    df = pd.DataFrame(data= scaling_vector,
                      index = np.array([i for i in range(len(scaling_vector))]),
                      columns = list(df.columns.values))

    descaled_df = pd.DataFrame(data=[mean_np, max_np],
                               index=np.array([i for i in range(2)]),
                               columns=list(df.columns.values))

    df["Crit"] = Crit_df
    descaled_df["Crit"] = Crit_df

    return df, scaled_array, descaled_df


def save_the_model(model, name):

    # serialize weights to HDF5
    model.save(f"model_h5_{name}.h5")
    # model.save("model")

    print("Saved model to disk")


def plot_styling_black():

    # plt.style.use('seaborn-bright')
    plt.style.use('dark_background')

    plt.gca().yaxis.grid(True, color='gray')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 20

    plt.rcParams['axes.labelsize'] = 18
    # plt.rcParams['axes.labelweight'] = 'bold'

    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    plt.rcParams['legend.fontsize'] = 15
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.numpoints'] = 2
    plt.rcParams['legend.framealpha'] = None
    plt.rcParams["legend.frameon"] = False
    # plt.rcParams['legend.edgecolor'] = 'black'
    plt.rcParams["legend.framealpha"] = 0.7
    plt.rcParams["legend.borderpad"] = 0.7
    plt.rcParams["legend.borderaxespad"] = 1

    plt.rcParams['figure.titlesize'] = 30

    # plt.tick_params(top='False', bottom='False', left='False', right='False', labelleft='False', labelbottom='True')

    for spine in plt.gca().spines.values():
        spine.set_visible(False)


def plotting(df, x, y):

    # df.plot(kind='scatter', x='x', y='y', linestyle='--', marker='o')

    fig, ax = plt.subplots()

    ax.scatter(x=df[x], y=df[y], linestyle='--', marker='o')

    plt.title(f'{x} vs {y}')

    # for i in range(0, len(df), 10):
    #
    #     txt = f"T_g: {df.iloc[i].loc['T_g']}, {df.iloc[i].loc['spindle_pos']}mm"
    #     ax.annotate(txt, (df.iloc[i].loc[x], df.iloc[i].loc[y]))

    # plt.legend(title='Generator Temperature [ºC]')

    plt.show()


def eval_metric(model, history, metric_name='loss'):

    '''
    Function to evaluate a trained model on a chosen metric.
    Training and validation metric are plotted in a
    line chart for each epoch.

    Parameters:
        history : model training history
        metric_name : loss or accuracy
    Output:
        line chart with epochs of x-axis and metric on
        y-axis
    '''

    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]
    e = range(1, len(metric) + 1)
    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.xlabel('Epoch number')
    plt.ylabel(metric_name)
    plt.title('Comparing training and validation ' + metric_name + ' for ' + model.name)
    plt.legend()
    plt.savefig(f'{metric_name}.png')
    plt.clf()

    print('Training and validation metric plots saved')


def model_playground(X_train, X_valid, y_train, y_valid, test_set):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    # # Patience = 10 - Wait until 10 epochs to see if the error keeps up
    # es = EarlyStopping(monitor='val_loss', mode='min', patience=25, min_delta=0)


    model.fit(X_train, y_train, epochs=500,
              verbose=0,
              batch_size=10,
              use_multiprocessing=True,
              workers=8,
              callbacks=[es],
              validation_data=(X_valid, y_valid))

    # plot_model(model, to_file='C:\\Users\\diogo\\Desktop\\inegi\\DATA\\Ejector_Results\\model.png')

    # print('All good')

    # test_y_predictions = model.predict(test_scaled)
    test_y_predictions = model.predict(test_set)

    return model, test_y_predictions


def GradientBoostingRegressor_hyper_params():

    estimators = np.arange(1, 150, 20)
    loss_type = ['ls', 'lad', 'huber', 'quantile']
    learn_rate = np.arange(.1, 1, .1)
    samples_split = np.arange(2, 15, 2)
    samples_leaf = np.arange(2, 15, 2)
    depth = np.arange(1, 6, 1)

    i = 0

    clfs_to_test = {}

    for ests in estimators:
        for lss in loss_type:
            for lrn_rate in learn_rate:
                for smpls_splt in samples_split:
                    for smpls_leaf in samples_leaf:
                        for dpth in depth:

                            clfs_to_test[i] = GradientBoostingRegressor(n_estimators=ests,
                                                                        loss=lss, max_depth=dpth,
                                                                        learning_rate=lrn_rate, min_samples_leaf=smpls_leaf,
                                                                        min_samples_split=smpls_splt)

                            i += 1

    # clf_2 = RandomForestRegressor()

    return clfs_to_test


def ADABoostRegressor_hyper_params():

    estimators = np.arange(1, 200, 10)
    loss_type = ['linear', 'square', 'exponential']
    learn_rate = np.arange(.1, 2, .1)

    i = 0

    clfs_to_test = {}

    for ests in estimators:
        for lss in loss_type:
            for lrn_rate in learn_rate:

                clfs_to_test[i] = AdaBoostRegressor(n_estimators=ests,
                                                    loss=lss,
                                                    learning_rate=lrn_rate)

                i += 1

    # clf_2 = RandomForestRegressor()

    return clfs_to_test



def ElasticNet_hyper_params():

    alpha = np.arange(.1, 1, .1)
    l1_ratio = np.arange(.1, 1, .1)
    fit_intercept = [True, False]
    normalize = [True, False]
    max_iter = np.arange(1, 2000, 100)
    warm_start = [True, False]
    positive = [True, False]
    selection = ['random', 'cyclic']

    i = 0

    clfs_to_test = {}

    for alph in alpha:
        for l_rt in l1_ratio:
            for ft_intrcpt in fit_intercept:
                for nrmlz in normalize:
                    for mx_itr in max_iter:
                        for wrm_strt in warm_start:
                            for pstv in positive:
                                for slctn in selection:

                                    clfs_to_test[i] = ElasticNet(alpha=alph, l1_ratio=l_rt, fit_intercept=ft_intrcpt,
                                                                 normalize=nrmlz, max_iter=mx_itr,
                                                                 warm_start=wrm_strt, positive=pstv, selection=slctn)

                                    i += 1

    return clfs_to_test


def choose_ML_alg():

    models = [RANSACRegressor(), HuberRegressor(), LinearRegression(), ElasticNet(), ElasticNetCV(), Lars(),
              Lasso(), LassoLars(), LassoLarsIC(), OrthogonalMatchingPursuit(), OrthogonalMatchingPursuitCV(),
              Ridge(), SGDRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), AdaBoostRegressor()]

    return models


def ML_Playground(X_train, X_test, y_train, y_test, test_set, scaled_array, clf):

    clf.fit(X_train, y_train)

    test_y_predictions = clf.predict(test_set)

    return clf, test_y_predictions

    # min_mnse = 999
    # mnse = mean_squared_error(y_test, test_y_predictions)
    # i = 0
    #
    #
    #
    # if mnse < min_mnse:
    #
    #     min_mnse = mnse
    #
    #     print(f'Trial {i}')
    #     print('MNSE: ', mnse)
    #     print('Mean Abs Error: ', mean_absolute_error(y_test, test_y_predictions))
    #     print('\n')
    #     print('-' * 20)
    #     print('-' * 20)
    #     print('\n')
    #
    # Tc = {}
    # Tg = {}
    # spindle_mm = {}
    # j = 0
    #
    # for T_c in range(26, 55):
    #     Tc[j] = T_c
    #     Tg[j] = 70
    #     spindle_mm[j] = 8
    #
    #     j += 1
    #
    # print('\n')
    #
    # test_set = pd.DataFrame({'T_c': Tc,
    #                          'T_g': Tg,
    #                          'spindle_pos': spindle_mm})
    #
    # # print(scaled_array)
    # # quit()
    #
    # test_scaled = (test_set - scaled_array['mean_np']) / scaled_array['max_np']
    #
    # test_y_predictions = clf.predict(test_scaled)
    #
    # ER = {}
    #
    # for i in range(len(test_y_predictions)):
    #     ER[i] = test_y_predictions[i]
    #
    # prediction_set = pd.DataFrame({'T_c': Tc,
    #                                'T_g': Tg,
    #                                'spindle_pos': spindle_mm,
    #                                'ER': ER})
    #
    # # predict_critical_temp(prediction_set)
    #
    # print(prediction_set)
    # plotting(prediction_set, 'T_c', 'ER')
    #
    # quit()


def get_double_chocking(df):

    print(df)

    unique_Tc = df.loc[:, 'T_c'].unique()
    unique_Tg = df.loc[:, 'T_g'].unique()
    unique_mm = df.loc[:, 'spindle_pos'].unique()

    for tg in unique_Tg:
        for mm in unique_mm:

            print('\n')
            unique_df = df.loc[df['T_g'] == tg].loc[df['spindle_pos'] == mm]
            print(unique_df)

    quit()


def save_to_csv(model, mnse, mean_error, date_today):

    ML_directory = f'C:\\Users\\diogo\\Desktop\\inegi\\ML_data\\'

    if not os.path.exists(ML_directory):
        os.makedirs(ML_directory)

    file_name = f'{ML_directory}{date_today}.csv'

    df = pd.DataFrame({'Model': pd.Series([model]),
                       'MNSE': pd.Series([mnse]),
                       'Mean_ERROR': pd.Series([mean_error])})

    # if file does not exist write header
    if not os.path.isfile(file_name):
        df.to_csv(file_name, index=False)

    else:  # else it exists so append without writing the header
        df.to_csv(file_name, mode='a', index=False, header=False)


def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


def swish(x, beta=1):
    return (x * sigmoid(beta * x))


def NN(train_X):

    # Read Heng-Tze Cheng 2016 paper - Wide & Deep

    input = keras.layers.Input(shape = train_X.shape[1:])

    # Dropout Layer
    # Dropout is one of the most popular regularization techniques for deep neural networks.
    # It was proposed23 by Geoffrey Hinton in 2012 and further detailed in a paper24
    # by Nitish Srivastava et al., and it has proven to be highly successful

    keras.layers.Dropout(0.5, noise_shape=None, seed=None)

    get_custom_objects().update({'Swish': Swish(swish)})

    # # Architecture 1
    # hidden1 = keras.layers.Dense(12, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm())(input)
    # hidden2 = keras.layers.Dense(12, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm())(hidden1)
    # hidden3 = keras.layers.Dense(12, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm())(hidden2)

    # # Architecture 2
    # hidden1 = keras.layers.Dense(20, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm())(input)
    # hidden2 = keras.layers.Dense(20, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm())(hidden1)
    # hidden3 = keras.layers.Dense(20, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm())(hidden2)

    #
    #   There are a few different weight constraints to choose from. A good simple constraint for this model is to simply normalize the weights so that the norm is equal to 1.0.
    #   This constraint has the effect of forcing all incoming weights to be small.
    #   unit_norm in Keras
    #

    # # Architecture 3
    # hidden1 = keras.layers.Dense(30, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm())(input)
    # hidden2 = keras.layers.Dense(30, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm())(hidden1)
    # hidden3 = keras.layers.Dense(30, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm())(hidden2)

    # Architecture 3
    hidden1 = keras.layers.Dense(30, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm(), kernel_regularizer=keras.regularizers.l2(0.01))(input)

    hidden2 = keras.layers.Dense(30, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm(), kernel_regularizer=keras.regularizers.l2(0.01))(hidden1)

    hidden3 = keras.layers.Dense(30, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm(), kernel_regularizer=keras.regularizers.l2(0.01))(hidden2)

    # # Architecture 4
    # hidden1 = keras.layers.Dense(50, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm())(input)
    # hidden2 = keras.layers.Dense(50, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm())(hidden1)
    # hidden3 = keras.layers.Dense(50, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm())(hidden2)

    concat = keras.layers.Concatenate()([input, hidden2, hidden3])

    # Regression: linear (because values are unbounded)
    output = keras.layers.Dense(2, activation='linear', kernel_initializer="he_normal", kernel_constraint=unit_norm())(concat)

    model = keras.models.Model(inputs = [input], outputs = [output])

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse'])

    # model.compile(loss = euclidean_distance_loss,
    #               optimizer='adam',
    #               metrics=['mse'])

    return model


def wide_and_deep_NN(train_X):

    # Read Heng-Tze Cheng 2016 paper - Wide & Deep

    input = keras.layers.Input(shape = train_X.shape[1:])

    # Dropout Layer
    keras.layers.Dropout(0.5, noise_shape=None, seed=None)

    get_custom_objects().update({'Swish': Swish(swish)})

    #
    #   There are a few different weight constraints to choose from. A good simple constraint for this model is to simply normalize the weights so that the norm is equal to 1.0.
    #   This constraint has the effect of forcing all incoming weights to be small.
    #   unit_norm in Keras
    #

    # Architecture 4
    hidden1 = keras.layers.Dense(40, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm())(input)
    hidden2 = keras.layers.Dense(40, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm())(hidden1)
    hidden3 = keras.layers.Dense(40, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm())(hidden2)

    # Wide layer
    wide = keras.layers.DenseFeatures(train_X.shape[1:], name='wide_inputs')(input)

    concat = keras.layers.Concatenate()([input, hidden2, hidden3])

    output = keras.layers.Dense(2)(concat)

    model = keras.models.Model(inputs = [input], outputs = [output])

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse'])

    return model




def NeuralNets(train_X):

    train_X = train_X.to_numpy().reshape(train_X.shape[0], 1, train_X.shape[1])

    model = Sequential()

    # model.add(LSTM(4, input_shape=train_X.shape[1:]))

    model.add(LSTM(64, input_shape=(train_X.shape[1:])))
    model.add(Dropout(0.5))

    model.add(LSTM(15, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(15, dropout=0.2, recurrent_dropout=0.2))

    model.add(Dense(2))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse'])

    return model





def model_playground(X_train, X_valid, y_train, y_valid, test_set):

    zero_values_test = test_set.index[test_set['Crit'] == 0].tolist()

    X_train = X_train.drop(['Crit'], axis=1)
    X_valid = X_valid.drop(['Crit'], axis=1)
    test_set = test_set.drop(['Crit'], axis=1)

    # scaler = StandardScaler()
    #
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_valid_scaled = scaler.transform(X_valid)

    model = NN(X_train)
    # model = NeuralNets(X_train)

    # Patience = 10 - Wait until 10 epochs to see if the error keeps up
    es = EarlyStopping(monitor='val_loss', mode='min', patience=25, min_delta=0)

    history = model.fit(X_train, y_train, epochs=500,
                        verbose = 0,
                        batch_size=10,
                        callbacks=[es],
                        validation_data=(X_valid, y_valid))

    # model.fit(X_train, y_train, epochs=500,
    #           verbose=0,
    #           batch_size=10,
    #           callbacks=[es],
    #           validation_data=(X_valid, y_valid))

    # plot_model(model, to_file='C:\\Users\\diogo\\Desktop\\inegi\\DATA\\Ejector_Results\\model.png')

    # print('All good')

    # test_y_predictions = model.predict(test_scaled)
    test_y_predictions = model.predict(test_set)

    test_set_ER = test_set.drop(zero_values_test)
    pred_ER = model.predict(test_set_ER)

    return model, test_y_predictions, history, pred_ER


def Class_NN(train_X):

    from keras.regularizers import l2

    # model = keras.models.Sequential([
    #     keras.layers.Flatten(input_shape=train_X.shape[1:]),
    #     keras.layers.Dense(30, activation="tanh", kernel_initializer="he_uniform", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)),
    #     keras.layers.Dense(30, activation="tanh", kernel_initializer="he_uniform", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)),
    #     keras.layers.Dense(1, activation="sigmoid")
    # ])

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=train_X.shape[1:]),
        keras.layers.Dense(50, activation="tanh", kernel_initializer="he_uniform", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(50, activation="tanh", kernel_initializer="he_uniform", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)),
        keras.layers.Dropout(rate=0.2),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    # Dropout Layer
    # Dropout is one of the most popular regularization techniques for deep neural networks.
    # It was proposed23 by Geoffrey Hinton in 2012 and further detailed in a paper24
    # by Nitish Srivastava et al., and it has proven to be highly successful

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics= ["acc"])

    # keras.metrics.Precision(name='precision')

    return model


def classification_playground(X_train, X_valid, test_set):

    # print(test_set.describe())

    y_train = X_train.loc[:,"Crit"].astype(int)
    y_valid = X_valid.loc[:, "Crit"].astype(int)

    X_train = X_train.drop(['Crit'], axis=1)
    X_valid = X_valid.drop(['Crit'], axis=1)
    test_set = test_set.drop(['Crit'], axis=1)

    # print(type(y_train))
    # print(y_valid.dtypes)
    #
    # quit()

    model = Class_NN(X_train)

    # Patience = 10 - Wait until 10 epochs to see if the error keeps up
    es = EarlyStopping(monitor='val_loss', mode='min', patience=25, min_delta=0)

    history = model.fit(X_train, y_train, epochs=500,
                        batch_size= 32,
                        callbacks=[es],
                        validation_data=(X_valid, y_valid))

    # evaluate the model
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_valid, y_valid, verbose=0)

    # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.show()

    # plot loss during training
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.show()


    # model.fit(X_train, y_train, epochs=500,
    #           verbose=0,
    #           batch_size=10,
    #           callbacks=[es],
    #           validation_data=(X_valid, y_valid))

    test_y_predictions = model.predict_classes(test_set)

    # print(test_y_predictions)
    # print(test_y_predictions.shape)
    # quit()

    return model, test_y_predictions, history


def main():

    date_start = datetime.now().strftime("%d_%m_%Y_%H_%M")

    # ejector_df = csv_func("Results_2020-06-25").astype(float)
    ejector_df = csv_func("Results_Classification").astype(float)
    # ejector_df = csv_func("Results_2020-07-06").astype(float)

    ejector_TST_df = csv_TST("Results_Classification").astype(float)
    # ejector_TST_df = csv_TST("Results_2020-07-06").astype(float)

    ejector_df_original = ejector_df

    # ejector_df = get_double_chocking(ejector_df_original)

    ejector_df = ejector_df_original.drop(['outlet'], axis=1)

    # print(ejector_df_original)

    # Get negative ER values to zero
    ejector_df['sec_mdot'] = ejector_df['sec_mdot'].clip(lower=0)
    ejector_TST_df['sec_mdot'] = ejector_df['sec_mdot'].clip(lower=0)

    # plotting(ejector_df, 'T_c', 'ER')

    # plotting(ejector_df, 'T_c', 'ER')

    try:
        ejector_df = ejector_df.drop(['ER'], axis=1)
        ejector_TST_df = ejector_TST_df.drop(['ER'], axis=1)

    except:
        pass

    try:
        ejector_df = ejector_df.drop(['outlet'], axis=1)
        ejector_TST_df = ejector_TST_df.drop(['outlet'], axis=1)

    except:
        pass

    cols = ejector_df.columns.tolist()

    cols = cols[-1:] + cols[:-1]

    ejector_df = ejector_df[cols]
    ejector_TST_df = ejector_TST_df[cols]

    ejector_df, scaled_array, descaled_df = feature_scaling(ejector_df)

    # Try to change variable to split the training and test sets - maybe spindle_pos
    X_train, y_train, X_valid, y_valid = define_trainig_set(ejector_df, "T_c")
    _X, _y, X_test, y_test = define_trainig_set(ejector_df, "T_c")

    Crit_df = ejector_TST_df.loc[:, "Crit"]
    ejector_TST_df = ejector_TST_df.drop(['Crit'], axis=1)

    TST_df_scaled = (ejector_TST_df - scaled_array['mean_np']) / scaled_array['std_np']

    TST_df_scaled["Crit"] = Crit_df

    features_TST_add = TST_df_scaled.drop(["prim_mdot", "sec_mdot"], axis=1)
    labels_TST_add = TST_df_scaled[["prim_mdot", "sec_mdot"]].copy()

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    X_test = pd.concat([X_test, features_TST_add]).reset_index(drop=True)
    y_test = pd.concat([y_test, labels_TST_add]).reset_index(drop=True)

    # print(features_TST_add)
    # print(labels_TST_add)
    # print(X_test)
    # print(y_test)
    #
    # quit()

    test_set = pd.DataFrame({'spindle_pos': X_test.loc[:,'spindle_pos'],
                             'Design_Tc': X_test.loc[:, 'Design_Tc'],
                             'Design_Tg': X_test.loc[:, 'Design_Tg'],
                             'Design_kW': X_test.loc[:, 'Design_kW'],
                             'T_c': X_test.loc[:,'T_c'],
                             'T_g': X_test.loc[:,'T_g'],
                             'Crit': X_test.loc[:,'Crit']})

    print('X_train: ', X_train.shape)
    print('y_train: ', y_train.shape)
    print('X_test: ', X_test.shape)
    print('y_test: ', y_test.shape)
    print('X_valid: ', X_valid.shape)
    print('y_valid: ', y_valid.shape)

    print('\n'*5)
    min_mnse = 999
    min_abs = 999
    i_best = 999

    min_mnse_ER = 999
    min_abs_ER = 999
    i_best_ER = 999
    best_model = ''

    min_mnse_CLASS = 999
    min_abs_CLASS = 999
    i_best_CLASS = 999

    # clfs_to_test = GradientBoostingRegressor_hyper_params()
    # clfs_to_test = ADABoostRegressor_hyper_params()
    clfs_to_test = choose_ML_alg()
    # clfs_to_test = ElasticNet_hyper_params()

    # for i in range(len(clfs_to_test)):
    #
    #     # model, test_y_predictions = ML_Playground(X_train, X_test, y_train, y_test, test_set, scaled_array, clfs_to_test[i])
    #     model, test_y_predictions = DL_Playground(X_train, X_test, y_train, y_test, test_set, scaled_array,clfs_to_test[i])
    #
    #     mnse = mean_squared_error(y_test, test_y_predictions)
    #     mean_error = mean_absolute_error(y_test, test_y_predictions)
    #
    #     save_to_csv(model, mnse, mean_error, date_start)
    #
    #     if mnse < min_mnse:
    #
    #         best_model = model
    #         min_mnse = mnse
    #         min_abs = mean_error
    #         i_best = i
    #
    #     print(f'{i}/{len(clfs_to_test)} - {mnse} - Best on {i_best}')

    print('\n'*10)

    # print('-'*20)
    # print('-' * 20, '\n')
    # print('BEST MODEL:')
    # print(best_model)
    # quit()

    y_test_CLASS = X_test.loc[:, "Crit"]

    model_classification, test_y_predictions_classification, history_classification = classification_playground(X_train, X_valid, test_set)

    mnse_CLASS = mean_squared_error(y_test_CLASS, test_y_predictions_classification)
    save_the_model(model_classification, "Critical_Tc_Classification")

    print(mnse_CLASS)
    quit()

    for i in range(200):

        model, test_y_predictions, history, predicted_ER = model_playground(X_train, X_valid, y_train, y_valid, test_set)

        print(f'Trial {i}/200')

        try:

            # DELETE HERE

            # zero_values_train = X_train.index[X_train['Crit'] == 0].tolist()
            # zero_values_valid = X_valid.index[X_valid['Crit'] == 0].tolist()
            zero_values_test = test_set.index[test_set['Crit'] == 0].tolist()

            # X_train_ER = X_train.drop(zero_values_train)
            # X_valid_ER = X_valid.drop(zero_values_valid)
            test_set_ER = test_set.drop(zero_values_test)

            y_test_ER = y_test.drop(zero_values_test)

            # predicted_ER = pd.DataFrame(test_y_predictions, columns=y_test_ER.columns, index=y_test_ER.index)
            # predicted_ER = test_y_predictions.drop(zero_values_test)
            # predicted_ER = np.delete(test_y_predictions, zero_values_test)

            # y_train_ER = y_train.drop(zero_values_train)
            # y_valid_ER = y_valid.drop(zero_values_valid)

            # UNTIL HERE

            mnse = mean_squared_error(y_test, test_y_predictions)
            mnse_ER = mean_squared_error(y_test_ER, predicted_ER)

            if mnse_ER < min_mnse_ER:

                print("NEW BEST ER MODEL")

                i_best_ER = i

                min_mnse_ER = mnse_ER
                min_abs_ER = mean_absolute_error(y_test_ER, predicted_ER)

                save_the_model(model, "ER_Regression")
                print("Model Saved")

            else:
                pass

            if mnse < min_mnse:

                print("NEW BEST MODEL")

                i_best = i

                min_mnse = mnse
                min_abs = mean_absolute_error(y_test, test_y_predictions)

                save_the_model(model, "General_Regression")
                print("Model Saved")


            else:
                pass


            if mnse_CLASS < min_mnse_CLASS:

                print("NEW BEST CLASSIFICATION MODEL")

                min_mnse_CLASS = mnse_CLASS
                min_abs_CLASS = mean_absolute_error(y_test_CLASS, test_y_predictions_classification)

                save_the_model(model_classification, "Critical_Tc_Classification")
                print("Model Saved")


            else:
                pass

            print('MNSE General: ', mnse)
            print('MNSE ER: ', mnse_ER)
            print('MNSE Classification: ', mnse_CLASS)
            print()
            print("----- General Model -----")
            print(f'Best on Trial {i_best}')
            print('MNSE: ', min_mnse)
            print('Mean Abs Error: ', min_abs)
            print()
            print("----- ER Model -----")
            print(f'Best on Trial {i_best_ER}')
            print('MNSE: ', min_mnse_ER)
            print('Mean Abs Error: ', min_abs_ER)
            print()
            print("----- Classification Model -----")
            print('MNSE: ', min_mnse_CLASS)
            print('Mean Abs Error: ', min_abs_CLASS)
            print('\n'*2)

        except:

            raise

    print('\n')
    print('-' * 20)
    print('-' * 20)
    print('\n')

    # test_scaled = (test_set - scaled_array['mean_np']) / scaled_array['max_np']


    Tc = {}
    Tg = {}
    spindle_mm = {}
    j = 0

    # for i in range(26, 47):
    #
    #     Tc[j] = i
    #     Tg[j] = 95
    #     spindle_mm[j] = 6
    #
    #     j += 1
    #
    #
    # print('\n')
    # test_set = pd.DataFrame({'T_c': Tc,
    #                          'T_g': Tg,
    #                          'spindle_pos': spindle_mm})

    # test_scaled = (test_set - scaled_array['mean_np']) / scaled_array['max_np']

    # test_y_predictions = model.predict(test_scaled)
    #
    # ER = {}
    #
    # for i in range(len(test_y_predictions)):
    #
    #     ER[i] = test_y_predictions[i][0]
    #
    #
    # prediction_set = pd.DataFrame({'T_c': Tc,
    #                                'T_g': Tg,
    #                                'spindle_pos': spindle_mm,
    #                                'ER': ER})
    #
    # print(prediction_set)
    # plotting(prediction_set, 'T_c', 'ER')
    #
    # quit()

    # Adicionar inicilizição

if __name__ == '__main__':

    plot_styling_black()
    see_all()
    np.seterr(divide='ignore', invalid='ignore')

    main()
