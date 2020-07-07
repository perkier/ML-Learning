import pandas as pd
import numpy as np
import os
import time


def get_other_directory(root_dir):

    for subdir, dirs, files in os.walk(root_dir):
        root_dict = dirs
        break

    root_arr = []

    for i in range(len(root_dict)):
        root_arr.append(os.path.join(root_dir, root_dict[i]))

    return root_arr



def critical_bP_loop(critical_bP_df, ref_ER_value):

    for i in range(1, len(critical_bP_df)):

        error = 100 * (ref_ER_value - critical_bP_df.iloc[i].loc['ER']) / ref_ER_value

        if abs(error) > 1.5:

            critical_range = f"[{critical_bP_df.iloc[i-1].loc['T_c']} - {critical_bP_df.iloc[i].loc['T_c']}]"

            break

    try:
        return critical_range

    except NameError:

        critical_range = f"[{int(critical_bP_df.iloc[i].loc['T_c'])}+]"

        # print(critical_bP_df, '\n',critical_range, abs(error),'\n'*2)

        return critical_range


def make_more_data(df, critical_range, T_g, mm, Times, Right_Augmentation = "No"):

    df = df.sort_values(by=['T_c'], ascending=True).reset_index(drop=True)

    print(len(df))
    print(df)

    low_Tc = int(df.iloc[0].loc['T_c'])

    T_min = 26
    T_max = 61

    if low_Tc > T_min:
        lowest_Tc = T_min

    elif low_Tc == T_min:
        lowest_Tc = T_min+1

    else:
        lowest_Tc = low_Tc

    try:
        critical_Tc = int(critical_range.split('[')[1].split(' -')[0])

    except:
        critical_Tc = int(critical_range.split('[')[1].split('+')[0])

    j = 0
    sum_ER = 0
    sum_p_mdot = 0
    sum_s_mdot = 0

    for i in range(lowest_Tc-2, critical_Tc+1):

        sum_df = df.loc[df['T_c'] == f'{i}'].reset_index(drop=True)

        if len(sum_df) > 0:
            sum_ER += float(sum_df.iloc[0].loc['ER'])
            sum_p_mdot += float(sum_df.iloc[0].loc['prim_mdot'])
            sum_s_mdot += float(sum_df.iloc[0].loc['sec_mdot'])
            j += 1

    augmented_df = pd.DataFrame({'T_c': [i/Times for i in range(lowest_Tc*Times, critical_Tc*Times)],
                                 'T_g': np.full((abs(critical_Tc - lowest_Tc)*Times), T_g),
                                 'spindle_pos': np.full((abs(critical_Tc - lowest_Tc)*Times), mm),
                                 'prim_mdot': np.full((abs(critical_Tc - lowest_Tc))*Times, sum_p_mdot/j),
                                 'sec_mdot': np.full((abs(critical_Tc - lowest_Tc)) * Times, sum_s_mdot / j),
                                 'ER': np.full((abs(critical_Tc - lowest_Tc))*Times, sum_ER/j),
                                 'Crit': np.full((abs(critical_Tc - lowest_Tc))*Times, 1)})

    unique_Tc = df.loc[:, 'T_c'].unique()

    for i in range(len(augmented_df)):

        for j in unique_Tc:

            if augmented_df.iloc[i].loc['T_c'] == j:
                augmented_df.drop([i])

    augmented_df = augmented_df.sort_values(by=['T_c'], ascending=True).reset_index(drop=True)

    high_Tc = df.loc[:,'T_c'].max()
    high_Tc_ER = df.loc[df['T_c'] == high_Tc].iloc[0].loc['ER']

    if Right_Augmentation == "Yes":

        if high_Tc_ER <= 0:

            high_Tc = int(df.loc[df['ER'] <= 0].iloc[0].loc['T_c'])

            if high_Tc > T_max:
                highest_Tc = T_min

            elif low_Tc == T_max:
                highest_Tc = T_max + 1

            else:
                highest_Tc = high_Tc

            # high_augm_df = pd.DataFrame({'T_c': [i for i in range(highest_Tc, T_max)],
            #                              'T_g': np.full(len(range(highest_Tc, T_max)), T_g),
            #                              'spindle_pos': np.full(len(range(highest_Tc, T_max)), mm),
            #                              'prim_mdot': np.full(len(range(highest_Tc, T_max)), df.loc[:,'prim_mdot'].mean()),
            #                              'sec_mdot': np.full(len(range(highest_Tc, T_max)), 0),
            #                              'ER': np.full(len(range(highest_Tc, T_max)), 0),
            #                              'Crit': np.full(len(range(highest_Tc, T_max)), 0)})

            high_augm_df = pd.DataFrame({'T_c': [i / Times for i in range(highest_Tc * Times, T_max * Times)],
                                         'T_g': np.full((abs(T_max - highest_Tc) * Times), T_g),
                                         'spindle_pos': np.full((abs(T_max - highest_Tc) * Times), mm),
                                         'prim_mdot': np.full((abs(T_max - highest_Tc) * Times), df.loc[:,'prim_mdot'].mean()),
                                         'sec_mdot': np.full((abs(T_max - highest_Tc) * Times), 0),
                                         'ER': np.full((abs(T_max - highest_Tc) * Times), 0),
                                         'Crit': np.full((abs(T_max - highest_Tc) * Times), 0)})

            # augmented_df = pd.DataFrame({'T_c': [i / 2 for i in range(lowest_Tc * 2, critical_Tc * 2)],
            #                              'T_g': np.full((abs(critical_Tc - lowest_Tc) * 2), T_g),
            #                              'spindle_pos': np.full((abs(critical_Tc - lowest_Tc) * 2), mm),
            #                              'prim_mdot': np.full((abs(critical_Tc - lowest_Tc)) * 2, sum_p_mdot / j),
            #                              'sec_mdot': np.full((abs(critical_Tc - lowest_Tc)) * 2, sum_s_mdot / j),
            #                              'ER': np.full((abs(critical_Tc - lowest_Tc)) * 2, sum_ER / j),
            #                              'Crit': np.full((abs(critical_Tc - lowest_Tc)) * 2, 1)})

            augmented_df = pd.concat([augmented_df, high_augm_df], ignore_index=True).drop_duplicates().reset_index(drop=True)

    else:
        pass

    return augmented_df


def get_critical_bP(df):

    unique_Tg = df.loc[:, 'T_g'].unique()

    Tg = {}
    spindle_pos = {}
    critical_range = {}
    i = 0

    augmented_df = pd.DataFrame()

    for T_g in unique_Tg:

        unique_df = df.loc[df['T_g'] == T_g].reset_index(drop=True)
        unique_spindle = unique_df.loc[:, 'spindle_pos'].unique()

        for mm in unique_spindle:

            critical_bP_df = unique_df.loc[unique_df['spindle_pos'] == mm].sort_values(by=['T_c'], ascending=True).reset_index(drop=True)
            ref_ER_value = critical_bP_df.iloc[0].loc['ER']

            if len(critical_bP_df) >= 2:

                two_rows_error = 100 * ( float(critical_bP_df.iloc[0].loc['ER']) - float(critical_bP_df.iloc[1].loc['ER']) ) / float(critical_bP_df.iloc[0].loc['ER'])

                if abs(two_rows_error) <= 1:

                    critical_range[i] = critical_bP_loop(critical_bP_df, ref_ER_value)

                    # augmented_to_merge = make_more_data(critical_bP_df, critical_range[i], T_g,mm, 2)
                    augmented_to_merge = make_more_data(critical_bP_df, critical_range[i], T_g, mm, 10, "Yes")

                    augmented_df = pd.concat([augmented_df, augmented_to_merge])

                    # if (i % 1) == 0:
                    #     augmented_df = pd.concat([augmented_df, augmented_to_merge])
                    #
                    # else:
                    #     pass

                else:
                    critical_range[i] = 'Need more Data'

            else:
                critical_range[i] = 'Need more Data'

            Tg[i] = T_g
            spindle_pos[i] = mm

            i += 1

    critical_df = pd.DataFrame({'T_g': Tg,
                                'spindle_pos': spindle_pos,
                                'Critical_Tc': critical_range}).sort_values(by=['T_g', 'spindle_pos'], ascending=True).reset_index(drop=True)

    augmented_df = augmented_df.reset_index(drop=True)

    try:

        concatenates = pd.concat([df, augmented_df], sort=True).sort_values(by=['T_c', 'T_g', 'spindle_pos'],ascending=True).reset_index(drop=True)
        concatenates = concatenates.drop(['iterations', 'mass_blnc[%]', 'prim_cnv_lst_10[%]', 'sec_cnv_lst_10[%]'], axis=1)
        concatenates = concatenates.apply(pd.to_numeric).sort_values(by=['T_g', 'spindle_pos', 'T_c'],ascending=True).reset_index(drop=True)

        # concatenates = concatenates.drop(['Crit'], axis=1)

    except:

        concatenates = pd.DataFrame()

    unique_Tg = concatenates.loc[:, 'T_g'].unique()

    final_concat = pd.DataFrame()

    for T_g in unique_Tg:

        unique_df = concatenates.loc[concatenates['T_g'] == T_g].reset_index(drop=True)

        unique_spindle = unique_df.loc[:, 'spindle_pos'].unique()

        for mm in unique_spindle:

            critical_bP_df = unique_df.loc[unique_df['spindle_pos'] == mm].sort_values(by=['T_c'], ascending=True).reset_index(drop=True)
            ref_ER_value = critical_bP_df.iloc[0].loc['ER']

            # print(critical_bP_df.loc[critical_bP_df["ER"] <= ref_ER_value*1.05])

            error = 5/100

            in_values = critical_bP_df[critical_bP_df["ER"].between(ref_ER_value*(1-error), ref_ER_value*(1+error))]
            out_values = critical_bP_df[critical_bP_df["ER"].between(-2, ref_ER_value * (1 - error))]

            in_values = in_values.assign(Crit=1)
            out_values = out_values.assign(Crit=0)

            final_concat = pd.concat([final_concat, in_values, out_values], sort=True).sort_values(by=['T_c', 'T_g', 'spindle_pos'],ascending=True).reset_index(drop=True)

            # print(ref_ER_value)
            # print(in_values)
            # print(out_values)
            # print('\n')

    concatenates = final_concat.copy()

    return critical_df, concatenates


def see_all():
    # Alongate the view on DataFrames

    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000000)

    # pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.max_colwidth', 1000)

    pd.set_option('float_format', '{:.10f}'.format)


def get_other_directory(root_dir):

    # root_dict = ['D:\\Diogo\\bioVGE\\Simulations\\5kW\\80_10_40__Spindle',
    #              'D:\\Diogo\\bioVGE\\Simulations\\5kW\\80_10_40__Spindle2',
    #              'D:\\Diogo\\bioVGE\\Simulations\\5kW\\80_10_40__Spindle3',
    #              'D:\\Diogo\\bioVGE\\Simulations\\5kW\\80_10_40__Spindle4',
    #              'D:\\Diogo\\bioVGE\\Simulations\\5kW\\80_10_40__Spindle5',
    #              'D:\\Diogo\\bioVGE\\Simulations\\5kW\\80_10_40__Ang12']

    for subdir, dirs, files in os.walk(root_dir):
        root_dict = dirs
        break

    root_arr = []

    for i in range(len(root_dict)):
        root_arr.append(os.path.join(root_dir, root_dict[i]))

    return root_arr


def directories_loop(rootdir):

    i = 0

    T_g = {}
    T_c = {}
    spindle_pos = {}
    files_repo = {}

    for subdir, dirs, files in os.walk(rootdir):

        for file in files:

            length = len(os.path.join(subdir, file).split('\\'))

            file_name = os.path.join(subdir, file).split('\\')[length - 1]

            if len(file_name.split('.txt')) > 1:

                splits = os.path.join(subdir, file).split('\\')

                if splits[len(splits) - 1] == "INFO.txt":

                    location = os.path.join(subdir, file)

                    csv_reader = open(location, 'rb')

                    try:
                        csv_read = pd.read_csv(csv_reader, encoding='utf-8', delimiter=',')

                    except:
                        print(location)
                        quit()

                    csv_reader.close()

                    conditions = list(csv_read)


            if len(file_name.split('.out')) > 1:

                splits = os.path.join(subdir, file).split('\\')

                OLD_1_0 = 0

                for part in splits:

                    if part == 'OLD':
                        T_c[i] = None

                    elif len(part.split('_')) > 1:

                        if part.split('_')[0] == 'Tc':

                            if OLD_1_0 == 0:
                                T_c[i] = part.split('_')[1]

                            else:
                                pass

                        elif part.split('_')[1] == 'mm':

                            if OLD_1_0 == 0:
                                spindle_pos[i] = part.split('_')[0]

                            else:
                                pass

                        elif part.split('_')[0] == 'Tg':

                            if OLD_1_0 == 0:
                                T_g[i] = part.split('_')[1]

                            else:
                                pass

                    else:
                        pass

                files_repo[i] = os.path.join(subdir, file)

                i += 1

    df = pd.DataFrame({'T_c': T_c,
                       'T_g': T_g,
                       'spindle_pos': spindle_pos,
                       'FN': files_repo}).dropna()

    return df, conditions


def get_mdot(location):

    csv_reader = open(location, 'rb')
    try:
        csv_read = pd.read_csv(csv_reader, encoding='utf-8', delimiter=' ')

    except:
        print(location)
        quit()

    csv_reader.close()

    df_column = list(csv_read)

    m_dot = float(csv_read.iloc[len(csv_read)-1].loc[df_column[0]])

    return m_dot


def get_iterations(location):

    csv_reader = open(location, 'rb')
    csv_read = pd.read_csv(csv_reader, encoding='utf-8', delimiter=' ')
    csv_reader.close()

    iteration = csv_read.iloc[len(csv_read)-1].name

    return iteration


def check_convergency(location, number):

    csv_reader = open(location, 'rb')
    csv_read = pd.read_csv(csv_reader, encoding='utf-8', delimiter=' ')
    csv_reader.close()

    df_column = list(csv_read)

    try:

        last_ten = csv_read.loc[:,df_column[0]].tail(number).astype(float).reset_index(drop=True)

        maximum = last_ten.describe().loc['max']
        minimum = last_ten.describe().loc['min']

        last_value = csv_read.loc[:, df_column[0]].tail(1).astype(float).reset_index(drop=True).iloc[0]

        diff = abs(maximum - minimum)
        diff_perc = 100*(diff / last_value)

        std = float(diff_perc)

    except:

        std = "std_ERROR"

    return std


def aglomerate_out_files(df):

    unique_Tc = df.loc[:,'T_c'].unique()
    unique_Tg = df.loc[:, 'T_g'].unique()
    unique_spindle = df.loc[:, 'spindle_pos'].unique()

    i = 0
    T_g_dict = {}
    T_c_dict = {}
    spindle_pos_dict = {}

    sec_location = {}
    prim_location = {}
    sec_convergency = {}
    prim_convergency = {}
    outlet_location = {}
    mass_balance = {}
    ER = {}
    iterations = {}

    convergency_number = 10

    for T_c in unique_Tc:

        for T_g in unique_Tg:

            for mm in unique_spindle:

                unique_df = df.loc[df['T_c'] == T_c]
                unique_df = unique_df.loc[df['T_g'] == T_g]
                unique_df = unique_df.loc[df['spindle_pos'] == mm].reset_index(drop=True)

                for j in range(len(unique_df)):

                    file_name = unique_df.iloc[j].loc['FN']
                    file_name_length = len(file_name.split('\\'))
                    file_name = file_name.split('\\')[file_name_length - 1]

                    out_splits = file_name.split('.out')[0].split('-')[0]

                    if out_splits == 'sec_inlet_mdot':

                        sec_location[i] = get_mdot(unique_df.iloc[j].loc['FN'])
                        sec_convergency[i] = check_convergency(unique_df.iloc[j].loc['FN'], convergency_number)

                    elif out_splits == 'primary_inlet_mdot':

                        prim_location[i] = get_mdot(unique_df.iloc[j].loc['FN'])
                        prim_convergency[i] = check_convergency(unique_df.iloc[j].loc['FN'], convergency_number)
                        iterations[i] = get_iterations(unique_df.iloc[j].loc['FN'])

                    elif file_name == 'report-def-0-rfile.out':

                        outlet_location[i] = get_mdot(unique_df.iloc[j].loc['FN'])

                    elif file_name == 'INFO.txt':

                        INFO = get_mdot(unique_df.iloc[j].loc['FN'])

                        print(INFO)

                        quit()

                try:
                    mass_balance[i] = 100 * abs(( float(outlet_location[i]) + float(prim_location[i]) + float(sec_location[i]) ) / float(outlet_location[i]))

                except:
                    mass_balance[i] = 'm_blnc_ERROR'

                T_g_dict[i] = T_g
                T_c_dict[i] = T_c

                if float(mm) >= 10:
                    spindle_pos_dict[i] = mm

                elif float(mm) < 10:
                    spindle_pos_dict[i] = f'0{mm}'


                try:
                    ER[i] = float(sec_location[i] / prim_location[i])

                except:
                    ER[i] = 'ER_ERROR'


                i += 1


    df = pd.DataFrame({'T_c': T_c_dict,
                       'T_g': T_g_dict,
                       'spindle_pos': spindle_pos_dict,
                       'ER': ER,
                       'prim_mdot': prim_location,
                       'sec_mdot': sec_location,
                       'mass_blnc[%]': mass_balance,
                       f'prim_cnv_lst_{convergency_number}[%]': prim_convergency,
                       f'sec_cnv_lst_{convergency_number}[%]': sec_convergency,
                       'iterations': iterations}).dropna().reset_index(drop=True)

    return df

def open_spindle_augmentation(df):

    unique_Tg = df.loc[:, 'T_g'].unique()
    unique_Tc = df.loc[:, 'T_c'].unique()

    # print(df)
    # print(df.dtypes)
    #
    # quit()


    for tg in unique_Tg:
        for tc in unique_Tc:

            unique_df = df.loc[df['T_g'] == tg].loc[df['T_c'] == tc].reset_index(drop=True)

            max_mm_string = unique_df.loc[:, 'spindle_pos'].max()

            try:
                max_mm = int(max_mm_string)

            except:
                max_mm = 0

            if max_mm > 20:

                ER = unique_df.loc[unique_df['spindle_pos'] == max_mm_string].reset_index(drop=True).iloc[0].loc['ER']
                p_mdot = unique_df.loc[unique_df['spindle_pos'] == max_mm_string].reset_index(drop=True).iloc[0].loc['prim_mdot']
                s_mdot = unique_df.loc[unique_df['spindle_pos'] == max_mm_string].reset_index(drop=True).iloc[0].loc['sec_mdot']

                to_add_df = pd.DataFrame({'T_c': np.full((abs(max_mm - 20)), tc),
                                          'T_g': np.full((abs(max_mm - 20)), tg),
                                          'spindle_pos': [i for i in range(20, max_mm)],
                                          'outlet': np.full((abs(max_mm - 20)), np.nan),
                                          'ER': np.full((abs(max_mm - 20)), ER),
                                          'prim_mdot': np.full((abs(max_mm - 20)), p_mdot),
                                          'sec_mdot': np.full((abs(max_mm - 20)), s_mdot),
                                          'mass_blnc[%]': np.full((abs(max_mm - 20)), np.nan),
                                          'prim_cnv_lst_10[%]': np.full((abs(max_mm - 20)), np.nan),
                                          'sec_cnv_lst_10[%]': np.full((abs(max_mm - 20)), np.nan),
                                          'iterations': np.full((abs(max_mm - 20)), np.nan)})

                df = pd.concat([df, to_add_df], sort=False).sort_values(by=['T_g', 'spindle_pos', 'T_c'], ascending=True).reset_index(drop=True)

            else:
                pass

    df = df.sort_values(by=['T_g', 'spindle_pos', 'T_c'], ascending=True).reset_index(drop=True)

    return df



def main():

    # ROOT = 'D:\\Diogo\\Simulations_Reports_1stOrd\\'
    ROOT = 'D:\\Diogo\\Simulations_Reports_TST\\'

    root_dict = get_other_directory(ROOT)

    general_df = pd.DataFrame()
    general_aug_df = pd.DataFrame()

    for rootdir in root_dict:

        df, conditions = directories_loop(rootdir)
        df = df.reset_index(drop=True)

        df = aglomerate_out_files(df).sort_values(by=['T_c', 'T_g', 'spindle_pos'], ascending=True).reset_index(drop=True)

        kW = conditions[0]
        Tg = conditions[1]
        Te = conditions[2]
        Tc = conditions[3]

        spindle_augmentation_df = open_spindle_augmentation(df)

        critical_TC, augmented_df = get_critical_bP(spindle_augmentation_df)

        augmented_df = augmented_df.sort_values(by=['T_g', 'spindle_pos'], ascending=False).reset_index(drop=True)

        augmented_df['Design_kW'] = kW
        augmented_df['Design_Tg'] = Tg
        augmented_df['Design_Tc'] = Tc

        df['Design_kW'] = kW
        df['Design_Tg'] = Tg
        df['Design_Tc'] = Tc

        general_df = pd.concat([general_df, df], ignore_index=True).drop_duplicates().reset_index(drop=True)
        general_aug_df = pd.concat([general_aug_df, augmented_df], ignore_index=True).drop_duplicates().reset_index(drop=True)


        print(rootdir)
        print(Tg, Tc, kW, conditions)

    general_aug_df['outlet'] = general_aug_df.loc[:, 'prim_mdot'] + general_aug_df.loc[:, 'sec_mdot']

    print('\n' * 2, f"Num. of Simulations: {len(general_df)}; Num. of Augmented Simulations: {len(general_aug_df)} - { format(100 * abs(len(general_aug_df) - len(general_df)) / len(general_df), '.2f')  }% gain")

    newdirectory = f'{rootdir}\\BackUp'

    if not os.path.exists(newdirectory):
        os.makedirs(newdirectory)

    date = time.strftime("%Y-%m-%d")

    path = f'{ROOT}\\Results_{date}.csv'

    general_aug_df.to_csv(path, index=False)


if __name__ == '__main__':

    see_all()
    main()
