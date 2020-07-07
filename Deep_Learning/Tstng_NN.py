import pandas as pd
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Activation, Conv2D
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras.backend as K
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_squared_error

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class Check_Convergency(object):

    def __init__(self, df):

        self.df = df

        self.df = self.df.sort_values(by=['T_c', 'T_g', 'spindle_pos'], ascending=True).reset_index(drop=True)

    def Back_Pressure(self):

        from check_convergence import get_critical_bP, insert_critical_column

        self.critical_TC, self.augmented_df = get_critical_bP(self.df)

        self.augmented_df = insert_critical_column(self.critical_TC, self.augmented_df)

        return self.critical_TC, self.augmented_df

    def COP(self):

        from check_convergence import get_COP

        self.COP_df = get_COP(self.df_w_outlet)

        return self.COP_df

    def number_simulations(self):

        print('\n' * 2,f"Num. of Simulations: {len(self.df)}; Num. of Augmented Simulations: {len(self.augmented_df)} - {format(100 * abs(len(self.augmented_df) - len(self.df)) / len(self.df), '.2f')}% gain")


def see_all():
    # Alongate the view on DataFrames

    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000000)


def plot_styling_black():

    plt.style.use('seaborn-bright')

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


def get_error(df, target_df):

    unique_Tc = df.loc[:,'T_c'].unique()
    target_df = target_df.loc[target_df['T_c'].isin(unique_Tc)].reset_index(drop=True)

    unique_Tc = target_df.loc[:,'T_c'].unique()
    df = df.loc[df['T_c'].isin(unique_Tc)].reset_index(drop=True)

    target_df["key"] = target_df["spindle_pos"] * target_df["T_c"] * target_df["T_g"]
    df["key"] = df["spindle_pos"] * df["T_c"] * df["T_g"]

    target_df["ER_Original"] = target_df["ER"]
    target_df = target_df.drop(['ER', "prim_mdot", "sec_mdot"], axis=1)

    df["Merge_key"] = df["spindle_pos"].astype(str) + df["T_c"].astype(str) + df["T_g"].astype(str)
    target_df["Merge_key"] = target_df["spindle_pos"].astype(str) + target_df["T_c"].astype(str) + target_df["T_g"].astype(str)

    # result_df = pd.concat([df, target_df], axis=1, join='inner')
    result_df = pd.merge(df, target_df, on='Merge_key').dropna()

    try:
        ER_error = check_ER_error(result_df)

    except:
        ER_error = {"Mean": 99999, "MSE": 99999}

    result_df.drop(result_df.columns.difference(['ER', 'ER_new', 'ER_Original']), 1, inplace=True)

    last_25perc = result_df.describe().loc["25%"].min()


    result_df = result_df.loc[result_df["ER_Original"] > last_25perc].loc[result_df["ER_new"] > last_25perc].loc[result_df["ER"] > last_25perc].reset_index(drop=True)

    mse_normal = mean_squared_error(result_df["ER_Original"], result_df["ER"])
    mse_new = mean_squared_error(result_df["ER_Original"], result_df["ER_new"])

    result_df["Error_normal"] = abs(result_df["ER_Original"] - result_df["ER"])
    result_df["Error_normal"] = result_df["Error_normal"] / result_df["ER_Original"]
    result_df["Error_normal_%"] = result_df["Error_normal"] * 100

    result_df["Error_new"] = abs(result_df["ER_Original"] - result_df["ER_new"])
    result_df["Error_new"] = result_df["Error_new"] / result_df["ER_Original"]
    result_df["Error_new_%"] = result_df["Error_new"] * 100

    error = {"mse_normal": mse_normal,
             "mse_new": mse_new,
             "relative_error_normal_%": result_df["Error_normal_%"].mean(),
             "relative_error_new_%": result_df["Error_new_%"].mean(),
             "ER Mean Error %": ER_error["Mean"],
             "ER MSE Error": ER_error["MSE"]}

    return error


def check_ER_error(df):

    df = df.loc[df["Crit"] == 1]
    df = df.loc[df["T_c_x"] <= 30]
    df = df.loc[df["ER_alg"] > 0].reset_index(drop=True)

    mse = mean_squared_error(df["ER_Original"], df["ER_alg"])

    ERROR = abs(df["ER_Original"] - df["ER_alg"])
    ERROR = ERROR / df["ER_Original"]
    ERROR_PERC = ERROR * 100

    ER_error = {"Mean": ERROR_PERC.mean(), "MSE": mse}

    return ER_error


def create_critical_plots(df, original_df, critic_TC_df, ER_type, Title):

    plt.clf()
    plot_styling_black()

    df = df.astype(float)
    original_df = original_df.astype(float)

    design_tc = df.iloc[0].loc['Design_Tc']
    design_tg = df.iloc[0].loc['Design_Tg']
    design_kW = df.iloc[0].loc['Design_kW']

    Tg = df.iloc[0].loc['T_g']

    target_df = original_df.loc[original_df['Design_Tc'] == design_tc]
    target_df = target_df.loc[target_df['Design_Tg'] == design_tg]
    target_df = target_df.loc[target_df['Design_kW'] == design_kW]
    target_df = target_df.loc[target_df['T_g'] == Tg].reset_index(drop=True)

    unique_spindle_pos = df.loc[:,'spindle_pos'].unique()

    target_df = target_df.loc[target_df['spindle_pos'].isin(unique_spindle_pos)]

    # check_ER_error(df, target_df)

    # quit()

    # df = df.loc[df['T_g'] == Tg].reset_index(drop=True)

    # df = df.loc[df['COP'] > 0].reset_index(drop=True)
    # df['COP%'] = df['COP'].astype(float).apply(lambda x: x * 100)

    df['ER%'] = df[ER_type].astype(float).apply(lambda x: x * 100)

    try:
        error = get_error(df, target_df)

    except:

        error = {'mse_normal': 99999999,
                 'mse_new': 99999999,
                 'relative_error_normal_%': 99999999,
                 'relative_error_new_%': 99999999}


    mm_30 = df.loc[df['spindle_pos'] > 20].index.values
    max_mm = df.loc[df['spindle_pos'] < 20].sort_values(by=['spindle_pos'], ascending=False).reset_index(drop=True).iloc[0].loc['spindle_pos'] + 1

    fig = plt.figure(1, figsize=(2500, 200))

    ax = fig.add_subplot(1,1,1)

    sc = ax.scatter(df['T_c'], df[ER_type], c=df['spindle_pos'].astype(int), cmap='jet', alpha=1, s=30)
    ax.scatter(target_df['T_c'], target_df['ER'], c=target_df['spindle_pos'].astype(int), cmap='jet', alpha=0.25,s=15)

    # ax.scatter(critic_TC_df['T_c'], np.full(len(critic_TC_df), 0.5), c=critic_TC_df['spindle_pos'].astype(int), cmap='jet', alpha=0.5, s=45, marker="*")


    cmap = sc.get_cmap()

    max_spindle_pos = unique_spindle_pos.max()
    min_spindle_pos = unique_spindle_pos.min()
    rel_max = max_spindle_pos - min_spindle_pos

    ax.scatter(critic_TC_df['T_c'], np.full(len(critic_TC_df), 0.9), c=cmap((critic_TC_df['spindle_pos'].astype(int) - min_spindle_pos) / rel_max), cmap='jet', alpha=0.5, s=45, marker="*")
    print(critic_TC_df)

    for mm in unique_spindle_pos:

        unique_df = df.loc[df['spindle_pos'] == mm].reset_index(drop=True)

        unique_tg = unique_df.loc[:,'T_g'].unique()

        for tg in unique_tg:

            unique_tg_df = unique_df.loc[unique_df['T_g'] == tg].reset_index(drop=True)

            if len(unique_tg_df) > 1:

                rgba = cmap((mm - min_spindle_pos) / rel_max)

                ax.plot(unique_tg_df['T_c'], unique_tg_df[ER_type], alpha=0.2, c=rgba, dashes=[6, 2])

                # save_csv(unique_df, f'{mm}mm_Tg{tg}')

    ax.set_xlim(30, 55)
    ax.set_ylim(0.1, 1)
    ax.set_xlabel('Condenser Temperature [ºC]', fontsize=8)
    ax.set_ylabel('Entrainment Ratio', fontsize=8)

    fig.colorbar(sc).set_label(label='Spindle Position [mm]', size=8)

    plt.title(f'Critical Condenser Temperature Curves for \n {Title}', fontsize=12)

    if error['mse_normal'] > 999:
        text = ''

    else:
        text = f" General Algorithm: \n Mean Squared Error  \n Normal: {'{:.2e}'.format(error['mse_normal'])} \n With Normalization: {'{:.2e}'.format(error['mse_new'])} \n \n Mean Error (Relative %) \n Normal: {round(error['relative_error_normal_%'], 2)} \n With Normalization: {round(error['relative_error_new_%'], 2)}  \n \n ER Algorithm: \n Mean Squared Error : {'{:.2e}'.format(error['ER MSE Error'])} \n Mean Error: {round(error['ER Mean Error %'], 2)}%"

    plt.text(42, 0.3, text, ha='left', wrap=True, fontsize=6)

    plt.savefig(f'D:\\Diogo\\Deep_Learning\\imgs\\Plots\\{Title}.png', dpi=300)

    # plt.show()

    # plt.show(block=False)
    #
    # fig = plt.figure(2, figsize=(12, 12))
    #
    # ax = fig.add_subplot(111)
    #
    # sc = ax.scatter(df['spindle_pos'], df['ER'], c=df['T_c'].astype(int), cmap='jet', alpha=1, s=30)
    # ax.scatter(target_df['spindle_pos'], target_df['ER'], c=target_df['T_c'].astype(int), cmap='jet', alpha=0.25,s=15)
    #
    # cmap = sc.get_cmap()
    #
    # unique_Tc = df.loc[:, 'T_c'].unique()
    #
    # max_Tc = unique_Tc.max()
    # min_Tc = unique_Tc.min()
    # rel_max = max_Tc - min_Tc
    #
    # for Tc in unique_Tc:
    #
    #     unique_df = df.loc[df['T_c'] == Tc].reset_index(drop=True)
    #
    #     if len(unique_df) > 1:
    #
    #         rgba = cmap((Tg - min_Tc) / rel_max)
    #
    #         ax.plot(unique_df['spindle_pos'], unique_df['ER'], alpha=0.2, c=rgba, dashes=[6, 2])
    #
    #         # save_csv(unique_df, f'{mm}mm_Tg{tg}')
    #
    # # ax.set_xlim(30, 60)
    # ax.set_ylim(0, None)
    # ax.set_xlabel('Spindle Position [mm]')
    # ax.set_ylabel('Entrainment Ratio')
    #
    # fig.colorbar(sc, label='Generator Temperature [ºC]')
    # plt.title(f'Critical Condenser Temperature Curves for each Spindle Position')
    #
    # # plt.savefig(f'D:\\Diogo\\DATA\\OUT\\plt1.png', dpi=300)
    #
    # plt.show()

    # quit()
    #
    # # save_csv(optimum_df, f'optimum_df')
    #
    #
    # fig = plt.figure(2, figsize=(12, 12))
    #
    # ax = fig.add_subplot(111)
    #
    # print(optimum_df.dtypes)
    # # quit()
    #
    # sc = ax.scatter(optimum_df['T_c'].astype(float), optimum_df['Q_g'], c=optimum_df['T_g'].astype(int), cmap='jet', alpha=1)
    # ax.scatter(optimum_df['T_c'].astype(float), optimum_df['Q_e'], c=optimum_df['T_g'].astype(int), cmap='jet', alpha=1)
    #
    # cmap = sc.get_cmap()
    #
    # unique_tg = optimum_df.loc[:, 'T_g'].astype(int).unique()
    #
    # max_tg = unique_tg.max()
    # min_tg = unique_tg.min()
    # rel_max = max_tg - min_tg
    #
    # for tg in unique_tg:
    #
    #     unique_tg_df = optimum_df.loc[optimum_df['T_g'] == f'{tg}'].reset_index(drop=True)
    #
    #     print(unique_tg_df)
    #
    #     if len(unique_tg_df) > 1:
    #
    #         rgba = cmap((tg - min_tg) / rel_max)
    #
    #         ax.plot(unique_tg_df['T_c'].astype(float), unique_tg_df['Q_g'], alpha=0.2, c=rgba, dashes=[6, 2])
    #         ax.plot(unique_tg_df['T_c'].astype(float), unique_tg_df['Q_e'], alpha=0.2, c=rgba, dashes=[6, 2])
    #
    #
    # ax.set_xlim(29, None)
    # ax.set_ylim(0, None)
    # ax.set_xlabel('Condenser Temperature [ºC]')
    # ax.set_ylabel('Power [W]')
    #
    # fig.colorbar(sc, label='Generator Temperature [ºC]')
    #
    # plt.title(f'Generator and Evaporator Power vs. Generator Temperature')
    # plt.savefig(f'D:\\Diogo\\DATA\\OUT\\plt2.png', dpi=300)
    #
    # plt.show()
    #
    # quit()




class Swish(Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'Swish'


def swish(x, beta=1):
    return (x * sigmoid(beta * x))


def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))



def import_model(name):

    # model_location = "D:\\Diogo\\py_scripts\\CODE\\model_h5.h5"
    model_location = f"D:\\Diogo\\Deep_Learning\\Models\\{name}.h5"

    # model_location = "D:\\Diogo\\py_scripts\\CODE\\model_h5_ER_Regression.h5"

    # model = load_model(model_location,
    #                    custom_objects = {'swish': Activation(swish),
    #                                      'euclidean_distance_loss': euclidean_distance_loss})

    # model = load_model(model_location,
    #                    custom_objects = {'Swish': Activation(swish)})

    model = load_model(model_location)

    return model


def csv_func(name = 'augmented_results'):

    csv_reader = open(f'D:\\Diogo\\Simulations_Reports_1stOrd\\{name}.csv', 'rb')
    csv_read = pd.read_csv(csv_reader, encoding='latin1', delimiter=',', dtype=np.float64)
    csv_reader.close()

    return csv_read


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


def predict_critical_temp(df):

    print('\n'*5)

    # from check_convergence import get_critical_bP
    # critical_df, concatenated = get_critical_bP(df)

    unique_tg = df.loc[:, 'T_g'].unique()
    unique_tc = df.loc[:, 'T_c'].unique()
    unique_mm = df.loc[:, 'spindle_pos'].unique()

    final_df = pd.DataFrame()

    # Group and divide results by Tg and Spindle Pos.
    for tg in unique_tg:

        unique_tg_df = df.loc[df['T_g'] == tg].reset_index(drop=True)

        for mm in unique_mm:

            unique_mm_df = unique_tg_df.loc[unique_tg_df['spindle_pos'] == mm].reset_index(drop=True)

            ER_0 = unique_mm_df.iloc[0].loc['ER']
            ER_new = {}

            j = 0

            # Do the calculations for each
            for i in range(len(unique_mm_df)):

                diff_perc = 100 * (ER_0 - unique_mm_df.iloc[i].loc['ER']) / ER_0
                # print(diff_perc)

                if abs(diff_perc) < 5:
                    # Do a mean value of the Entrainment Ratio
                    ER_0 = (ER_0*i + unique_mm_df.iloc[i].loc['ER']) / (i+1)

                else:
                    j += 1

                    if j == 1:
                        for z in range(i):
                            ER_new[z] = ER_0

                    ER_new[i] = unique_mm_df.iloc[i].loc['ER']

            unique_mm_df["ER_new"] = pd.Series(ER_new)

            final_df = pd.concat([final_df, unique_mm_df], ignore_index=True).drop_duplicates().reset_index(drop=True)

    return final_df


def visual_DL(model):

    plot_model(model, to_file='D:\\Diogo\\Deep_Learning\\imgs\\model_plot.png', show_shapes=True, show_layer_names=True)



def critical_Tc_alg(df, class_results):

    df["Crit"] = class_results

    # print(df)

    df = df.loc[df["Crit"] == 1]

    # quit()

    unique_Tg = df.loc[:, 'T_g'].unique()

    critical_df = []
    # critical_df = pd.DataFrame()

    for Tg in unique_Tg:

        unique_Tg_df = df.loc[df["T_g"] == Tg]

        unique_mm = unique_Tg_df.loc[:, 'spindle_pos'].unique()

        for mm in unique_mm:

            unique_mm_df = unique_Tg_df.loc[df["spindle_pos"] == mm].sort_values(by=['T_c'], ascending=False).reset_index(drop=True)

            critical_df.append(unique_mm_df.head(1))

    final_df = pd.concat(critical_df).reset_index(drop=True)

    return final_df


def main():

    Conditions = [[9, 80, 40],
                  [5, 80, 40],
                  [4, 80, 37],
                  [6.7, 80, 40],
                  [9, 85, 40],
                  [7.5, 80, 40],
                  [9, 80, 37]]

    Generator_Temp = [80, 90]

    # Conditions = [[6.7, 80, 40]]

    # Design_Tc = Conditions[0][2]
    # Design_Tg = Conditions[0][1]
    # Design_kW = Conditions[0][0]
    #
    # print(Design_Tc, Design_Tg, Design_kW)
    #
    # quit()

    for i in range(len(Conditions)):

        for Gen_Temp in Generator_Temp:

            Design_Tc = Conditions[i][2]
            Design_Tg = Conditions[i][1]
            Design_kW = Conditions[i][0]

            # Design_Tc = 37
            # Design_Tg = 80
            # Design_kW = 8

            # Gen_Temp = 80
            # Cond_Temp = 40

            Title = f"Design Conditions of {Design_Tg}[Tg];{Design_Tc}[Tc];{Design_kW}[kW] - Tg={Gen_Temp}ºC"

            ejector_df = csv_func("Results_2020-07-06").astype(float)
            Classification_df = csv_func("Results_Classification").astype(float)


            ejector_df['sec_mdot'] = ejector_df['sec_mdot'].clip(lower=0)
            Classification_df['sec_mdot'] = Classification_df['sec_mdot'].clip(lower=0)

            try:
                ejector_df = ejector_df.drop(['ER'], axis=1)
                Classification_df = Classification_df.drop(['ER'], axis=1)

            except:
                pass

            try:
                ejector_df = ejector_df.drop(['outlet'], axis=1)
                Classification_df = Classification_df.drop(['outlet'], axis=1)

            except:
                pass

            cols = ejector_df.columns.tolist()

            cols = cols[-1:] + cols[:-1]

            ejector_df = ejector_df[cols]
            Classification_df = Classification_df[cols]

            original_df = ejector_df.copy()
            original_df["ER"] = original_df["sec_mdot"] / original_df["prim_mdot"]

            original_Classification_df = Classification_df.copy()
            original_Classification_df["ER"] = original_Classification_df["sec_mdot"] / original_Classification_df["prim_mdot"]

            ejector_df, scaled_array, descaled_df = feature_scaling(ejector_df)
            Classification_df, Classification_scaled_array, Classification_descaled_df = feature_scaling(Classification_df)

            Tc = {}
            Tg = {}
            spindle_mm = {}
            j = 0

            model_ER = import_model("30_30_30_ER_Regression")
            model_class = import_model("30_30_30_Critical_Tc_Classification")
            model_general = import_model("30_30_30_General_Regression")

            visual_DL(model_general)

            for T_c in range(26, 55):
                for mm in range(4, 11, 2):

                    Tc[j] = T_c
                    Tg[j] = Gen_Temp
                    spindle_mm[j] = mm

                    j += 1

            print('\n')

            test_set = pd.DataFrame({'spindle_pos': pd.Series(spindle_mm),
                                     'Design_Tc': np.full((j), Design_Tc),
                                     'Design_Tg': np.full((j), Design_Tg),
                                     'Design_kW': np.full((j), Design_kW),
                                     'T_c': pd.Series(Tc),
                                     'T_g': pd.Series(Tg)})

            test_scaled = (test_set - scaled_array['mean_np'][:-2]) / scaled_array['std_np'][:-2]
            Classification_test_scaled = (test_set - Classification_scaled_array['mean_np'][:-2]) / Classification_scaled_array['std_np'][:-2]

            test_y_predictions = model_general.predict(test_scaled)
            test_y_predictions_ER = model_ER.predict(test_scaled)
            test_y_predictions_CLASS = model_class.predict_classes(Classification_test_scaled)

            # print(test_y_predictions_CLASS)
            # quit()

            # prim_mdot = [i for i in test_y_predictions.T[0]]
            # sec_mdot = [i for i in test_y_predictions.T[1]]

            m_dot_scaled = pd.DataFrame({'prim_mdot': [i for i in test_y_predictions.T[0]],
                                         'sec_mdot': [i for i in test_y_predictions.T[1]]})

            mdot_descaled = (m_dot_scaled * scaled_array['std_np'][-2:]) + scaled_array['mean_np'][-2:]

            m_dot_scaled_ER = pd.DataFrame({'prim_mdot_ER_alg': [i for i in test_y_predictions_ER.T[0]],
                                            'sec_mdot_ER_alg': [i for i in test_y_predictions_ER.T[1]]})

            mdot_descaled_ER = (m_dot_scaled_ER * scaled_array['std_np'][-2:]) + scaled_array['mean_np'][-2:]

            final_df = pd.concat([test_set, mdot_descaled, mdot_descaled_ER], axis=1, sort=False)

            final_df["ER"] = final_df["sec_mdot"] / final_df["prim_mdot"]
            final_df["ER_alg"] = final_df["sec_mdot_ER_alg"] / final_df["prim_mdot_ER_alg"]

            final_df.drop(["prim_mdot_ER_alg", "sec_mdot_ER_alg"], axis=1)

            final_df["ER"] = final_df["ER"].clip(lower=0)
            final_df["ER_alg"] = final_df["ER_alg"].clip(lower=0)

            final_df = predict_critical_temp(final_df)
            critic_TC_df = critical_Tc_alg(test_set, test_y_predictions_CLASS)

            final_df["ER_new"] = final_df["ER_new"].clip(lower=0)

            create_critical_plots(final_df, original_df, critic_TC_df, "ER_new", Title)
            # create_critical_plots(final_df, original_df, "ER", Title)


    # plotting(final_df, 'T_c', 'ER')

    # ejector_map = Check_Convergency(prediction_set)
    # critical_TC, augmented_df = ejector_map.Back_Pressure()
    # ejector_cop = ejector_map.COP()

    # print('\n'*10)
    # print(ejector_map)
    # print('\n')
    # print(critical_TC)
    # # print(ejector_cop)
    # print('\n')
    # ejector_map.number_simulations()

    # quit()

    # Criar teste para critical back-Pressure e teste para valores de Entrainment ratio no geral


if __name__ == '__main__':

    plot_styling_black()
    see_all()
    main()
