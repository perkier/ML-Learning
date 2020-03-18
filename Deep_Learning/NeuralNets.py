import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt


from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation, Conv2D
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras.backend as K
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.callbacks import EarlyStopping, History

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor


def csv_func(name = 'augmented_results'):

    csv_reader = open(f'C:\\Users\\diogo\\Desktop\\inegi\\DATA\\OUT\\{name}.csv', 'rb')
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


def splitting(df):

    split = StratifiedShuffleSplit(n_splits=1, test_size= 0.2, random_state=42)

    for train_index, test_index in split.split(df, df["T_g"]):

        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]

    return strat_train_set, strat_test_set


def define_trainig_set(df):

    strat_train_set, strat_test_set = splitting(df)

    savings_train = strat_train_set.drop('ER', axis=1)
    savings_labels_train = strat_train_set['ER'].copy()

    savings_test = strat_test_set.drop('ER', axis=1)
    savings_labels_test = strat_test_set['ER'].copy()

    X = df.drop('ER', axis=1)                       #Features
    y = df.loc[:,'ER'].copy()                       #Labels

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



def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'


def swish(x, beta=1):
    return (x * sigmoid(beta * x))


def neural_net(train_X):

    # create model
    model = Sequential()

    # get number of columns in training data
    n_cols = train_X.shape[1]

    # add model layers
    # model.add(Flatten(input_shape=[n_cols,]))       # Creating the first layer and add it to the model - Flatten Layer - Preprocesssing

    model.add(Dense(n_cols, activation='relu', input_shape=(n_cols,)))

    # model.add(Dense(33, activation='relu'))         # First hidden layer (10 neurons) - Bias already included (one per neuron)
    # model.add(Dense(33, activation='relu'))         # Second hidden layer (10 neurons) - Bias already included (one per neuron)

    get_custom_objects().update({'Swish': Activation(swish)})

    # Architecture 1
    model.add(Dense(33, activation='Swish'))  # First hidden layer (33 neurons) - Bias already included (one per neuron)
    model.add(Dense(33, activation='Swish'))  # Second hidden layer (33 neurons) - Bias already included (one per neuron)

    # # Architecture 2
    # model.add(Dense(33, activation='Swish'))    # First hidden layer (33 neurons) - Bias already included (one per neuron)
    # model.add(Dense(50, activation='Swish'))    # Second hidden layer (50 neurons) - Bias already included (one per neuron)
    # model.add(Dense(20, activation='Swish'))    # Third hidden layer (20 neurons) - Bias already included (one per neuron)

    # # Architecture 3
    # model.add(Dense(2, activation='Swish'))    # First hidden layer (33 neurons) - Bias already included (one per neuron)
    # model.add(Dense(5, activation='Swish'))    # Second hidden layer (50 neurons) - Bias already included (one per neuron)
    # model.add(Dense(2, activation='Swish'))    # Third hidden layer (20 neurons) - Bias already included (one per neuron)

    # Architecture 4
    # model.add(Dense(6, activation='Swish'))    # First hidden layer (33 neurons) - Bias already included (one per neuron)


    model.add(Dense(1, activation='sigmoid'))                              # Dense output layer (1 neurons - one per class)

    model.compile(loss = euclidean_distance_loss,
                  optimizer='adam',
                  metrics=['mse'])


    return model


def NN(train_X):

    # Read Heng-Tze Cheng 2016 paper - Wide & Deep

    input = keras.layers.Input(shape = train_X.shape[1:])

    get_custom_objects().update({'Swish': Swish(swish)})

    # # Architecture 1
    # hidden1 = keras.layers.Dense(33, activation='Swish')(input)
    # hidden2 = keras.layers.Dense(33, activation='Swish')(hidden1)
    # concat = keras.layers.Concatenate()([input, hidden2])

    # # Architecture 2
    # hidden1 = keras.layers.Dense(33, activation='Swish')(input)
    # hidden2 = keras.layers.Dense(50, activation='Swish')(hidden1)
    # hidden3 = keras.layers.Dense(20, activation='Swish')(hidden2)
    # concat = keras.layers.Concatenate()([input, hidden2, hidden3])

    #
    #   There are a few different weight constraints to choose from. A good simple constraint for this model is to simply normalize the weights so that the norm is equal to 1.0.
    #   This constraint has the effect of forcing all incoming weights to be small.
    #   unit_norm in Keras
    #

    # Architecture 3
    hidden1 = keras.layers.Dense(33, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm())(input)
    hidden2 = keras.layers.Dense(33, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm())(hidden1)
    hidden3 = keras.layers.Dense(33, activation='relu', kernel_initializer="he_normal", kernel_constraint=unit_norm())(hidden2)
    concat = keras.layers.Concatenate()([input, hidden2, hidden3])

    output = keras.layers.Dense(1)(concat)

    model = keras.models.Model(inputs = [input], outputs = [output])

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse'])

    # model.compile(loss = euclidean_distance_loss,
    #               optimizer='adam',
    #               metrics=['mse'])

    return model



def CNN(train_X):

    # create model
    model = Sequential()

    # get number of columns in training data
    n_cols = train_X.shape[1]

    model.add(Conv2D(33, kernel_size=3, activation='relu', input_shape=(n_cols,)))

    # model.add(Dense(33, activation='relu'))         # First hidden layer (10 neurons) - Bias already included (one per neuron)
    # model.add(Dense(33, activation='relu'))         # Second hidden layer (10 neurons) - Bias already included (one per neuron)

    get_custom_objects().update({'Swish': Activation(swish)})

    # Architecture 1
    model.add(Conv2D(33, kernel_size=3, activation='Swish'))  # First hidden layer (33 neurons) - Bias already included (one per neuron)
    model.add(Conv2D(33, kernel_size=3, activation='Swish'))  # Second hidden layer (33 neurons) - Bias already included (one per neuron)

    # # Architecture 2
    # model.add(Dense(33, activation='Swish'))    # First hidden layer (33 neurons) - Bias already included (one per neuron)
    # model.add(Dense(50, activation='Swish'))    # Second hidden layer (50 neurons) - Bias already included (one per neuron)
    # model.add(Dense(20, activation='Swish'))    # Third hidden layer (20 neurons) - Bias already included (one per neuron)

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))                              # Dense output layer (1 neurons - one per class)

    model.compile(loss = euclidean_distance_loss,
                  optimizer='adam',
                  metrics=['mse'])


    return model



def feature_scaling(df):

    brands = df.loc[:, 'ER']
    df = df.drop(columns='ER')

    df_np = df.to_numpy()

    mean_df = df.describe().loc['mean']
    mean_np = mean_df.to_numpy()

    max_df = df.describe().loc['max']
    max_np = max_df.to_numpy()

    min_df = df.describe().loc['min']
    min_np = min_df.to_numpy()

    max_np = max_np - min_np

    scaling_vector = (df_np - mean_np) / max_np

    scaled_array = {'instructions': '(df_np - mean_np) / max_np; T_c, T_g, spindle_pos', 'mean_np': mean_np, 'max_np': max_np}

    df = pd.DataFrame(data= scaling_vector,
                      index = np.array([i for i in range(len(scaling_vector))]),
                      columns = list(df.columns.values))

    descaled_df = pd.DataFrame(data=[mean_np, max_np],
                              index=np.array([i for i in range(2)]),
                              columns=list(df.columns.values))

    df = df.join(brands)

    return df, scaled_array, descaled_df


def save_the_model(model):

    # serialize model to JSON

    # model_json = model.to_json("model.json")
    #
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)

    # serialize weights to HDF5
    model.save("model.h5")
    model.save("model")

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

    # model = neural_net(X_train)
    model = NN(X_train)
    # model = CNN(X_train)

    # Patience = 10 - Wait until 10 epochs to see if the error keeps up
    es = EarlyStopping(monitor='val_loss', mode='min', patience=25, min_delta=0)

    # history = model.fit(X_train, y_train, epochs=500,
    #                     verbose = 0,
    #                     batch_size=10,
    #                     use_multiprocessing=True,
    #                     workers=8,
    #                     callbacks=[es],
    #                     validation_data=(X_valid, y_valid))

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

    # print(test_set)
    # print(test_y_predictions)
    # print(datetime.datetime.now())
    #
    # print(test_y_predictions)

    # return model, test_y_predictions, history
    return model, test_y_predictions


def ML_Playground(X_train, X_test, y_train, y_test, test_set, scaled_array):

    # clf = GradientBoostingRegressor(n_estimators=1,
    #                                 loss='ls', alpha=0.95, max_depth=3,
    #                                 learning_rate=.1, min_samples_leaf=9,
    #                                 min_samples_split=9)

    clf = RandomForestRegressor()
    # clf = GradientBoostingRegressor(n_estimators=75)

    clf.fit(X_train, y_train)

    test_y_predictions = clf.predict(test_set)

    min_mnse = 999
    mnse = mean_squared_error(y_test, test_y_predictions)
    i = 0

    if mnse < min_mnse:

        min_mnse = mnse

        print(f'Trial {i}')
        print('MNSE: ', mnse)
        print('Mean Abs Error: ', mean_absolute_error(y_test, test_y_predictions))
        print('\n')
        print('-' * 20)
        print('-' * 20)
        print('\n')

    Tc = {}
    Tg = {}
    spindle_mm = {}
    j = 0

    for T_c in range(26, 55):
        Tc[j] = T_c
        Tg[j] = 70
        spindle_mm[j] = 8

        j += 1

    print('\n')

    test_set = pd.DataFrame({'T_c': Tc,
                             'T_g': Tg,
                             'spindle_pos': spindle_mm})

    # print(scaled_array)
    # quit()

    test_scaled = (test_set - scaled_array['mean_np']) / scaled_array['max_np']

    test_y_predictions = clf.predict(test_scaled)

    ER = {}

    for i in range(len(test_y_predictions)):
        ER[i] = test_y_predictions[i]

    prediction_set = pd.DataFrame({'T_c': Tc,
                                   'T_g': Tg,
                                   'spindle_pos': spindle_mm,
                                   'ER': ER})

    # predict_critical_temp(prediction_set)

    print(prediction_set)
    plotting(prediction_set, 'T_c', 'ER')

    quit()


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



def main():

    print('\n'*10)

    ejector_df_Te10 = csv_func("augmented_15Evap_results")
    ejector_df_Te15 = csv_func("augmented_10Evap_results")

    ejector_df_original = merge_dfs(ejector_df_Te10, ejector_df_Te15)

    # ejector_df = get_double_chocking(ejector_df_original)

    ejector_df = ejector_df_original.drop(['outlet', 'Critical'], axis=1)

    # print(ejector_df_original)
    # quit()

    # ejector_df = ejector_df.loc[ejector_df['T_g'] >= 61].reset_index(drop=True)

    # Get negative ER values to zero
    ejector_df['ER'] = ejector_df['ER'].clip(lower=0)

    # ejector_df = ejector_df.loc[ejector_df['ER'] >= 0].reset_index(drop=True)

    # ejector_df = create_zero_values(ejector_df)

    # plotting(ejector_df, 'T_c', 'ER')

    # plotting(ejector_df, 'T_c', 'ER')

    ejector_df, scaled_array, descaled_df = feature_scaling(ejector_df)

    try:
        ejector_df = ejector_df.drop(['prim_mdot', 'sec_mdot', 'mass_blnc', 'prim_cnv_lst_10', 'sec_cnv_lst_10', 'iterations', 'outlet'], axis=1)

    except:
        pass

    try:
        ejector_df = ejector_df.drop(['outlet', 'Critical'], axis=1)

    except:
        pass

    print(ejector_df.head())

    # X_train, y_train, X_test, y_test = define_trainig_set(ejector_df)
    X_train, y_train, X_valid, y_valid = define_trainig_set(ejector_df)

    print('X_train: ', X_train.shape)
    print('y_train: ', y_train.shape)
    print('X_valid: ', X_valid.shape)
    print('y_valid: ', y_valid.shape)
    # quit()

    other_set, X_test_oblig, y_test_oblig = shape_critical_set(ejector_df_original)

    _X_1, _y_1, X_test, y_test = define_trainig_set(other_set)

    X_test = X_test.append(X_test_oblig)
    y_test = y_test.append(y_test_oblig)

    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    test_set = pd.DataFrame({'T_c': X_test.loc[:,'T_c'],
                             'T_g': X_test.loc[:,'T_g'],
                             'T_e': X_test.loc[:, 'T_e'],
                             'spindle_pos': X_test.loc[:,'spindle_pos']})

    # print(X_test)
    # print(y_test)
    #
    # quit()

    print('\n'*5)
    min_mnse = 999
    min_abs = 999
    i_best = 999

    # ML_Playground(X_train, X_test, y_train, y_test, test_set, scaled_array)

    for i in range(200):

        # model, test_y_predictions, history = model_playground(X_train, X_valid, y_train, y_valid, test_set)
        model, test_y_predictions = model_playground(X_train, X_valid, y_train, y_valid, test_set)

        try:

            mnse = mean_squared_error(y_test, test_y_predictions)

            if mnse < min_mnse:

                min_mnse = mnse
                min_abs = mean_absolute_error(y_test, test_y_predictions)
                save_the_model(model)
                i_best = i

                # eval_metric(model, history)
                # eval_metric(model, history, 'accuracy')

                # print('YEAHHHH')
                # eval_metric(model, history)
                # print('YEAHHHH*2')

            else:
                pass

            print(f'Trial {i}')
            print('MNSE: ', mnse)
            print()
            print(f'Best on Trial {i_best}')
            print('MNSE: ', min_mnse)
            print('Mean Abs Error: ', min_abs)
            print('\n')
            print('-' * 20)
            print('-' * 20)
            print('\n')

        except:

            pass



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
    main()
