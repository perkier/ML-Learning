import pandas as pd
import numpy as np
from datetime import date, datetime
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import RANSACRegressor, HuberRegressor, LinearRegression, ElasticNet, ElasticNetCV, Lars, Lasso, LassoLars, LassoLarsIC, OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib

from ngboost import NGBRegressor
from ngboost.distns import Normal



class Pre_Processing(object):

    def __init__(self):

        self.date_start = datetime.now().strftime("%d_%m_%Y_%H_%M")

        ejector_df_Te10 = self.csv_func("augmented_15Evap_results")
        ejector_df_Te15 = self.csv_func("augmented_10Evap_results")

        self.ejector_df_original = self.merge_dfs(ejector_df_Te10, ejector_df_Te15)

    def get_feats_labels(self):

        # Get negative ER values to zero
        self.ejector_df_original['ER'] = self.ejector_df_original['ER'].clip(lower=0)

        ejector_df = self.ejector_df_original.drop(['outlet', 'Critical'], axis=1)

        ejector_df, self.scaled_array, descaled_df = self.feature_scaling(ejector_df)

        try:
            ejector_df = ejector_df.drop(['prim_mdot', 'sec_mdot', 'mass_blnc', 'prim_cnv_lst_10', 'sec_cnv_lst_10', 'iterations', 'outlet'],axis=1)

        except:
            pass

        try:
            ejector_df = ejector_df.drop(['outlet', 'Critical'], axis=1)

        except:
            pass

        X_train, y_train, X_valid, y_valid = self.define_trainig_set(ejector_df)

        other_set, X_test_oblig, y_test_oblig = self.shape_critical_set()
        # shape_zero_set(ejector_df_original)


        _X_1, _y_1, X_test, y_test = self.define_trainig_set(other_set)

        X_test = X_test.append(X_test_oblig)
        y_test = y_test.append(y_test_oblig)

        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        test_set = pd.DataFrame({'T_c': X_test.loc[:, 'T_c'],
                                 'T_g': X_test.loc[:, 'T_g'],
                                 'T_e': X_test.loc[:, 'T_e'],
                                 'spindle_pos': X_test.loc[:, 'spindle_pos']})

        return ejector_df, X_train, X_test, y_train, y_test, test_set


    def csv_func(self, name = 'augmented_results'):

        csv_reader = open(f'C:\\Users\\diogo\\Desktop\\inegi\\DATA\\OUT\\{name}.csv', 'rb')
        csv_read = pd.read_csv(csv_reader, encoding='latin1', delimiter=',', dtype=np.float64)
        csv_reader.close()

        return csv_read


    def merge_dfs(self, ejector_df_Te10, ejector_df_Te15):

        ejector_df_Te10['T_e'] = pd.Series(np.full(len(ejector_df_Te10), float(10)))
        ejector_df_Te15['T_e'] = pd.Series(np.full(len(ejector_df_Te15), float(15)))

        ejector_concated = pd.concat([ejector_df_Te10, ejector_df_Te15]).sort_values(by=['T_e', 'T_g', 'spindle_pos'],ascending=True).reset_index(drop=True)

        return ejector_concated


    def feature_scaling(self, df):

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

        scaled_array = {'instructions': '(df_np - mean_np) / max_np; T_c, T_g, spindle_pos', 'mean_np': mean_np,
                        'max_np': max_np}

        df = pd.DataFrame(data=scaling_vector,
                          index=np.array([i for i in range(len(scaling_vector))]),
                          columns=list(df.columns.values))

        descaled_df = pd.DataFrame(data=[mean_np, max_np],
                                   index=np.array([i for i in range(2)]),
                                   columns=list(df.columns.values))

        df = df.join(brands)

        pickle.dump(scaled_array, open("C:\\Users\\diogo\\Desktop\\inegi\\ML_data\\scaled_array.p", "wb"))

        return df, scaled_array, descaled_df


    def splitting(self, df):

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        for train_index, test_index in split.split(df, df["T_g"]):
            strat_train_set = df.loc[train_index]
            strat_test_set = df.loc[test_index]

        return strat_train_set, strat_test_set


    def define_trainig_set(self, df):

        strat_train_set, strat_test_set = self.splitting(df)

        savings_train = strat_train_set.drop('ER', axis=1)
        savings_labels_train = strat_train_set['ER'].copy()

        savings_test = strat_test_set.drop('ER', axis=1)
        savings_labels_test = strat_test_set['ER'].copy()

        X = df.drop('ER', axis=1)  # Features
        y = df.loc[:, 'ER'].copy()  # Labels

        X_train = savings_train
        y_train = savings_labels_train
        X_test = savings_test
        y_test = savings_labels_test

        return X_train, y_train, X_test, y_test


    def shape_critical_set(self):

        df = self.ejector_df_original

        df = df.loc[df['spindle_pos'] <= 20].reset_index(drop=True)

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


class ML(object):

    def __init__(self):

        pass


    class Choose_params:

        def choose_ML_alg(self):

            models = [RANSACRegressor(), HuberRegressor(), LinearRegression(), ElasticNet(), ElasticNetCV(), Lars(),
                      Lasso(), LassoLars(), LassoLarsIC(), OrthogonalMatchingPursuit(), OrthogonalMatchingPursuitCV(),
                      Ridge(), SGDRegressor(), RandomForestRegressor(), GradientBoostingRegressor(),
                      AdaBoostRegressor(),
                      NGBRegressor(Dist=Normal), DecisionTreeRegressor()]

            return models


        def GradientBoostingRegressor_hyper_params(self):

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

            return clfs_to_test


        def ADABoostRegressor_hyper_params(self):

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

            return clfs_to_test


        def ElasticNet_hyper_params(self):

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
                                            clfs_to_test[i] = ElasticNet(alpha=alph, l1_ratio=l_rt,
                                                                         fit_intercept=ft_intrcpt,
                                                                         normalize=nrmlz, max_iter=mx_itr,
                                                                         warm_start=wrm_strt, positive=pstv,
                                                                         selection=slctn)

                                            i += 1

            return clfs_to_test


        def Lasso_hyper_params(self):

            alpha = np.arange(.1, 1, .1)
            fit_intercept = [True, False]
            normalize = [True, False]
            max_iter = np.arange(1, 2000, 100)
            warm_start = [True, False]
            positive = [True, False]
            selection = ['random', 'cyclic']

            i = 0

            clfs_to_test = {}

            for alph in alpha:
                for ft_intrcpt in fit_intercept:
                    for nrmlz in normalize:
                        for mx_itr in max_iter:
                            for wrm_strt in warm_start:
                                for pstv in positive:
                                    for slctn in selection:
                                        clfs_to_test[i] = Lasso(alpha=alph, fit_intercept=ft_intrcpt,
                                                                normalize=nrmlz, max_iter=mx_itr,
                                                                warm_start=wrm_strt, positive=pstv, selection=slctn)

                                        i += 1

            return clfs_to_test


pp = Pre_Processing()

ejector_df, X_train, X_test, y_train, y_test, test_set = pp.get_feats_labels()

