import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import ConvexHull


def see_all():
    # Alongate the view on DataFrames

    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 200)
    pd.set_option('display.width', 1000000)


def separate_by(df, parameter):
    # Create dict with various Dfs by selected parameter

    # Get unique parameters of df
    unique_params = df.loc[:,parameter].unique()

    param_dict = {}

    for param in unique_params:

        # Selects only the datapoints with specific parameter - e.g. Selects only datapoints with Nodes=90
        param_dict[param] = df.loc[df[parameter] == param].reset_index(drop=True)

    return param_dict


def corr_plot(df, nodes):

    # Prepare Data

    df.dropna(inplace=True)

    min_x = round(df.min().min() - 0.2, 2)
    max_x = round(df.max().max() + 0.2, 2)

    x = df.loc[:, ['Sec_mdot_Crit_Error[%]']]
    df['error_z'] = (x - x.mean()) / x.std()

    df['colors_1'] = ['red' if x < 0 else 'green' for x in df["Sec_mdot_Crit_Error[%]"]]
    df['colors_2'] = ['red' if x < 0 else 'green' for x in df["Prim_mdot_Crit_Error[%]"]]
    df['colors_3'] = ['red' if x < 0 else 'green' for x in df["ER_MNSE"]]
    df['colors_4'] = ['red' if x < 0 else 'green' for x in df["ER_Mean_Error"]]
    df['colors_5'] = ['red' if x < 0 else 'green' for x in df["General_MNSE"]]

    df.sort_values('error_z', inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Hyper_Param'}, inplace=True)

    # Draw plot
    plt.figure(figsize=(30, 15), dpi=80)

    plt.hlines(y=df.index, xmin=0, xmax=df.loc[:,"Sec_mdot_Crit_Error[%]"], color=df.colors_1, alpha=0.7, linewidth=10)

    for x, y, tex in zip(df.loc[:,"Sec_mdot_Crit_Error[%]"], df.index, df.Hyper_Param):
        t = plt.text(x, y, "Sec_mdot_Crit_Error[%]", horizontalalignment='right' if x < 0 else 'left',
                     verticalalignment='center', fontdict={'color': 'red' if x < 0 else 'green', 'size': 8})

    plt.hlines(y=df.index + 0.07, xmin=0, xmax=df.loc[:, "Prim_mdot_Crit_Error[%]"], color=df.colors_2, alpha=0.7, linewidth=10)

    for x, y, tex in zip(df.loc[:,"Prim_mdot_Crit_Error[%]"], df.index, df.Hyper_Param):
        t = plt.text(x, y + 0.07, "Prim_mdot_Crit_Error[%]", horizontalalignment='right' if x < 0 else 'left',
                     verticalalignment='center', fontdict={'color': 'red' if x < 0 else 'green', 'size': 8})

    plt.hlines(y=df.index - 0.07, xmin=0, xmax=df.loc[:, "ER_MNSE"], color=df.colors_3, alpha=0.7, linewidth=10)

    for x, y, tex in zip(df.loc[:,"ER_MNSE"], df.index, df.Hyper_Param):
        t = plt.text(x, y - 0.07, "ER_MNSE", horizontalalignment='right' if x < 0 else 'left',
                     verticalalignment='center', fontdict={'color': 'red' if x < 0 else 'green', 'size': 8})


    plt.hlines(y=df.index + 0.14, xmin=0, xmax=df.loc[:, "ER_Mean_Error"], color=df.colors_4, alpha=0.7, linewidth=10)

    for x, y, tex in zip(df.loc[:,"ER_Mean_Error"], df.index, df.Hyper_Param):
        t = plt.text(x, y + 0.14, "ER_Mean_Error", horizontalalignment='right' if x < 0 else 'left',
                     verticalalignment='center', fontdict={'color': 'red' if x < 0 else 'green', 'size': 8})

    plt.hlines(y=df.index - 0.14, xmin=0, xmax=df.loc[:, "General_MNSE"], color=df.colors_5, alpha=0.7, linewidth=10)

    for x, y, tex in zip(df.loc[:,"General_MNSE"], df.index, df.Hyper_Param):
        t = plt.text(x, y - 0.14, "General_MNSE", horizontalalignment='right' if x < 0 else 'left',
                     verticalalignment='center', fontdict={'color': 'red' if x < 0 else 'green', 'size': 8})

    # Decorations
    plt.gca().set(ylabel='$Hyper Parameter$', xlabel='$Error Correlation$')
    plt.yticks(df.index, df.Hyper_Param, fontsize=12)

    if nodes == 0:
        plt.title('Error Correlation with Hyper Parameter used to Train NN', fontdict={'size': 20})

    else:
        plt.title(f'Error Correlation with Hyper Parameter used to Train NN with {nodes} Nodes', fontdict={'size': 20})

    plt.grid(linestyle='--', alpha=0.5)
    plt.xlim(min_x , max_x)

    name = f"Error_Correlation_{nodes}Nodes"

    plt.savefig(f'D:\\Diogo\\Deep_Learning\\imgs\\Error_plots\\{name}.png', dpi=300)
    # plt.show(block=False)


def get_error_correlation(df, columns, plot, nodes=0):

    correlation_matrix = df.corr()
    correlation_matrix = correlation_matrix[columns].drop(columns, axis=0)

    if plot != "No":
        corr_plot(correlation_matrix, nodes)

    # overall_view = correlation_matrix.T.describe().drop(["count", "std"], axis=0)
    #
    # print(overall_view)

    # quit()


def get_Slope_Chart(df, param_1_to_compare, param_2_to_compare, error_comp, nodes):

    df = df.sort_values(by=[param_1_to_compare, param_2_to_compare], ascending=True).reset_index(drop=True)

    unique_params_1 = df.loc[:,param_1_to_compare].unique()

    stacked_df = pd.DataFrame()

    fig, ax = plt.subplots(1, 1, figsize=(14, 14), dpi=80)

    for param_1 in unique_params_1:

        selected_df = df.loc[df[param_1_to_compare] == param_1].reset_index(drop=True)
        unique_params_2 = selected_df.loc[:, param_2_to_compare].unique()

        param_1_column_mean = []
        param_1_column_min = []
        left_column = []

        for param_2 in unique_params_2:

            name = f"Do_{param_2}-Bsize_{param_1}"

            mean_value = selected_df.loc[selected_df[param_2_to_compare] == param_2].loc[:, error_comp].mean()
            min_value = selected_df.loc[selected_df[param_2_to_compare] == param_2].loc[:, error_comp].min()

            param_1_column_mean.append(mean_value)
            param_1_column_min.append(min_value)
            left_column.append(param_2)

        stacked_df[param_1] = param_1_column_mean
        stacked_df[param_2_to_compare] = left_column

        ax.plot(left_column, param_1_column_mean)

        # Vertical Lines
        ax.vlines(x=left_column, ymin=0, ymax=stacked_df.max().max() * 1.4, color='black', alpha=0.7, linewidth=1,
                  linestyles='dotted')
        # ax.vlines(x=left_column, ymin=0, ymax=stacked_df.max().max() * 1.4, color='black', alpha=0.7, linewidth=1,
        #           linestyles='dotted')

        # Points
        ax.scatter(y=param_1_column_mean, x=left_column, s=30, color='black', alpha=0.4)

    # print(stacked_df)
    #
    # quit()
    #
    # left_label = [str(c) + ', ' + str(round(y)) for c, y in zip(stacked_df[param_2_to_compare], df[32])]
    # right_label = [str(c) + ', ' + str(round(y)) for c, y in zip(stacked_df[param_2_to_compare], df[256])]
    # klass = ['red' if (y1 - y2) < 0 else 'green' for y1, y2 in zip(stacked_df[32], df[256])]

    # draw line
    # https://stackoverflow.com/questions/36470343/how-to-draw-a-line-with-matplotlib/36479941
    # def newline(p1, p2, color='black'):
    #     ax = plt.gca()
    #     l = mlines.Line2D([p1[0], p2[0]], [p1[1], p2[1]], color='red' if p1[1] - p2[1] > 0 else 'green', marker='o',
    #                       markersize=6)
    #     ax.add_line(l)
    #     return l


    # print(stacked_df)
    #
    # for
    #
    # ax.plot(t, s)

    # quit()

    # # Vertical Lines
    # ax.vlines(x=1, ymin=0, ymax=stacked_df.max().max()*1.4, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
    # ax.vlines(x=3, ymin=0, ymax=stacked_df.max().max()*1.4, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



    # Line Segmentsand Annotation
    # for p1, p2, c in zip(stacked_df[32], stacked_df[256], stacked_df[param_2_to_compare]):

        # newline([1, p1], [3, p2])
        # ax.text(1 - 0.05, p1, c + ', ' + str(round(p1, 2)), horizontalalignment='right', verticalalignment='center',
        #         fontdict={'size': 14})
        # ax.text(3 + 0.05, p2, c + ', ' + str(round(p2, 2)), horizontalalignment='left', verticalalignment='center',
        #         fontdict={'size': 14})

    # 'Before' and 'After' Annotations
    # ax.text(1 - 0.05, 13000, 'BEFORE', horizontalalignment='right', verticalalignment='center',
    #         fontdict={'size': 18, 'weight': 700})
    # ax.text(3 + 0.05, 13000, 'AFTER', horizontalalignment='left', verticalalignment='center',
    #         fontdict={'size': 18, 'weight': 700})

    # Decoration
    ax.set_title(f"{error_comp} behaviour with {param_1_to_compare} and {param_2_to_compare} changes With {nodes} Nodes", fontdict={'size': 17})
    ax.set(xlim=((stacked_df.loc[:,param_2_to_compare].min()*0.9), (stacked_df.loc[:,param_2_to_compare].max()*1.1)), ylim=(0, stacked_df.max().max()), ylabel=error_comp, xlabel=error_comp)
    # ax.set_xticks([1, 3])
    # ax.set_xticklabels(unique_params_2)
    # plt.yticks(np.arange(500, 13000, 2000), fontsize=12)

    # Lighten borders
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["bottom"].set_alpha(.0)
    plt.gca().spines["right"].set_alpha(.0)
    plt.gca().spines["left"].set_alpha(.0)

    name = f"{error_comp}_{param_1_to_compare}_{param_2_to_compare}_{nodes}_Nodes"

    plt.savefig(f'D:\\Diogo\\Deep_Learning\\imgs\\Error_plots\\{name}.png', dpi=300)
    # plt.show(block=False)


def dot_plot(df):

    # Draw horizontal lines
    fig, ax = plt.subplots(figsize=(16, 10), dpi=80)
    # ax.hlines(y=df.Nodes, xmin=0, xmax=100, color='gray', alpha=0.5, linewidth=.5, linestyles='dashdot')

    btch_colors = {32: 'tab:red', 64: 'tab:green', 128: 'tab:blue', 256: 'tab:orange'}
    df['Batch_Size_Color'] = df.Batch_Size.map(btch_colors)
    df["Crit_float_prim_mdot"] = df['Prim_mdot_Crit_Error[%]']/100
    df["Crit_float_prim_mdot"] = df["Crit_float_prim_mdot"].round(1)

    ax.scatter(y=df['Nodes'], x=df['Sec_mdot_Crit_Error[%]'], s=(df['Prim_mdot_Crit_Error[%]']**2)/2, alpha= 0.7, c=df['Batch_Size_Color'])

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels,title='Batch Size')

    plt.yticks(np.arange(30, 160, 10))

    name = f"Dot_Plot_Sec_mdot_Crit_Error[%]"

    plt.savefig(f'D:\\Diogo\\Deep_Learning\\imgs\\Error_plots\\{name}.png', dpi=300)
    # plt.show(block=False)


def cluster_plot(df):

    error_param = "Sec_mdot_Crit_Error[%]"
    # sec_axis = "Nodes"
    sec_axis = "Prim_mdot_Crit_Error[%]"

    n_clusters = 7

    # Agglomerative Clustering
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    # cluster.fit_predict(df[['Batch_Size', 'Dropout_Rate', 'Nodes', error_param]])
    cluster.fit_predict(df[[sec_axis, error_param]])

    plt.figure(1, figsize=(30, 15), dpi=80)

    # Encircle
    def encircle(x, y, ax=None, **kw):
        if not ax: ax = plt.gca()
        p = np.c_[x, y]
        hull = ConvexHull(p)
        poly = plt.Polygon(p[hull.vertices, :], **kw)
        ax.add_patch(poly)

    colors = ["gold", "tab:blue", "tab:red", "tab:green", "tab:orange", 'darkkhaki', 'palevioletred', 'deepskyblue', 'violet', 'peru','turquoise', 'springgreen', 'khaki', 'violet', 'deepskyblue', 'violet', 'peru', 'c', 'lime']

    best_cluster_error = 999

    for i in range(n_clusters):

        error_cluster = df.loc[cluster.labels_ == i].loc[:,"Sec_mdot_Crit_Error[%]"].describe()["25%"] + df.loc[cluster.labels_ == i].loc[:,"Prim_mdot_Crit_Error[%]"].describe()["25%"]
        # error_cluster = df.loc[cluster.labels_ == i].loc[:, "Sec_mdot_Crit_Error[%]"].min() + df.loc[cluster.labels_ == i].loc[:,"Prim_mdot_Crit_Error[%]"].min()

        if error_cluster < best_cluster_error:
            alpha_i = i
            best_cluster_error = error_cluster

        # print(i, df.loc[cluster.labels_ == i].loc[:,"Sec_mdot_Crit_Error[%]"].mean())

    # Draw polygon surrounding vertices
    for i in range(n_clusters):

        if i == alpha_i:
            encircle(df.loc[cluster.labels_ == i, error_param], df.loc[cluster.labels_ == i, sec_axis], ec="k", fc=colors[i], alpha=0.7, linewidth=0)

        else:
            encircle(df.loc[cluster.labels_ == i, error_param], df.loc[cluster.labels_ == i, sec_axis], ec="k", fc=colors[i], alpha=0.2, linewidth=0)

    # Plot Points
    plt.scatter(df.loc[:, error_param], df.loc[:, sec_axis], c=cluster.labels_, cmap='tab10')

    # Decorations
    plt.xlabel(error_param)
    plt.xticks(fontsize=12)
    plt.ylabel(sec_axis)
    plt.yticks(fontsize=12)
    plt.title('Cluster The Errors', fontsize=22)
    plt.savefig(f'D:\\Diogo\\Deep_Learning\\imgs\\Error_plots\\Cluster_Errors.png', dpi=300)
    # plt.show(block=False)

    # Create Plot with best Cluster Only
    plt.figure(2, figsize=(30, 15), dpi=80)

    df_best_cluster = df.loc[cluster.labels_ == alpha_i].reset_index(drop=True)

    df_best_cluster["joined_1"] = (df_best_cluster["Nodes"] ** (1 / df_best_cluster["Batch_Size"])) / df_best_cluster["Dropout_Rate"]
    df_best_cluster["joined_2"] = df_best_cluster["Nodes"] / df_best_cluster["Batch_Size"]

    # print(df_best_cluster.corr().loc[["joined_1", "joined_2", "Nodes", "Batch_Size", "Dropout_Rate"]])

    # Plot Points
    # plt.scatter(df_best_cluster[error_param], df_best_cluster[sec_axis], c=df_best_cluster['joined_2'], cmap='tab10')
    plt.scatter(df_best_cluster[error_param], df_best_cluster[sec_axis], c=df_best_cluster['Nodes'], cmap='tab10')

    # Decorations
    plt.xlabel(error_param)
    plt.xticks(fontsize=12)
    plt.ylabel(sec_axis)
    plt.yticks(fontsize=12)
    plt.title('Decide the Title', fontsize=22)
    plt.savefig(f'D:\\Diogo\\Deep_Learning\\imgs\\Error_plots\\Undecided_1.png', dpi=300)
    # plt.show(block=False)

    # Aglomerate Points by Nodes
    unique_nodes = df_best_cluster.loc[:,"Nodes"].unique()
    unique_batch = df_best_cluster.loc[:, "Batch_Size"].unique()
    unique_Do = df_best_cluster.loc[:, "Dropout_Rate"].unique()

    plt.figure(3, figsize=(30, 15), dpi=80)

    # Do_colors = {0.1: 'tab:red', 0.2: 'tab:green', 0.3: 'tab:blue'}
    Do_colors = {0.1: 'salmon', 0.2: 'springgreen', 0.3: 'skyblue'}
    # df_best_cluster['Do_Color'] = df_best_cluster.Dropout_Rate.map(Do_colors)

    df_aglom = df_best_cluster.groupby(["Nodes", "Batch_Size", "Dropout_Rate"]).count()
    # df_aglom = df_aglom.loc[:,[error_param]]
    df_aglom = df_aglom[error_param]

    # df_aglom["Color"] = df_best_cluster['Do_Color']

    # width = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6]
    width = [3, 2.5, 2,  1,0.5,0, -1, -1.5, -2, -3, -3.5, -4]

    fig, ax = plt.subplots()

    for node in unique_nodes:

        i = 0

        for b_size in unique_batch:

            for Do in unique_Do:

                j = 0

                try:
                    ax.bar(node - width[i], df_aglom[node][b_size][Do], 0.5, color=Do_colors[Do], label='Men')


                except:
                    pass

                if Do == 0.2:
                    ax.annotate(f'Batch Size {b_size}',
                                # xy=(node - width[i], df_aglom[node][b_size][Do]),
                                xy=(node - width[i], 0),
                                rotation=90,
                                size=10,
                                xytext=(0, 0),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

                j += 1
                i += 1

    plt.savefig(f'D:\\Diogo\\Deep_Learning\\imgs\\Error_plots\\Undecided_2.png', dpi=300)
    # plt.show(block=False)

    plt.figure(4, figsize=(30, 15), dpi=80)
    fig, ax = plt.subplots()

    cmap = plt.get_cmap("tab20c")
    # first_colors = cmap(np.arange(4) * 12)
    first_colors = ['darkkhaki', 'palevioletred', 'deepskyblue', 'violet', 'peru', 'palevioletred', 'deepskyblue', 'violet', 'peru','turquoise', 'springgreen', 'khaki', 'violet', 'deepskyblue', 'violet', 'peru', 'c', 'lime']
    outer_colors = cmap(np.arange(4) * 4)
    inner_colors = cmap([1, 2, 5])

    # ax.pie(df_aglom, radius=1 - 0.3, colors=outer_colors,
    #        wedgeprops=dict(width=0.3, edgecolor='w'))

    df_aglom_1 = df_best_cluster.groupby(["Nodes"])
    df_aglom_2 = df_best_cluster.groupby(["Nodes", "Batch_Size"])
    df_aglom_3 = df_best_cluster.groupby(["Nodes", "Batch_Size", "Dropout_Rate"])


    wedges, texts = ax.pie(df_aglom_1.Nodes.count(), radius=1+0.3, colors=first_colors,
                           wedgeprops=dict(width=0.3, edgecolor='w'),
                           startangle=0)

    ax.pie(df_aglom_2.Batch_Size.count(), radius=1, colors=outer_colors,
           wedgeprops=dict(width=0.3, edgecolor='w'))

    ax.pie(df_aglom_3.Dropout_Rate.count(), radius=1-0.3, colors=inner_colors,
           wedgeprops=dict(width=0.3, edgecolor='w'))

    ax.set(aspect="equal", title='Pie plot with `ax.pie`')

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(f"{unique_nodes[i]} Nodes", xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)

    plt.savefig(f'D:\\Diogo\\Deep_Learning\\imgs\\Error_plots\\Undecided_3.png', dpi=300)
    # plt.show(block=False)


def main():

    # Read Error Dataframe - Generated from ML_Parameter_Tunning.py
    error_df = pd.read_csv("D:\\Diogo\\Deep_Learning\\Error_Architecture\\CSVs\\error.csv", encoding='utf-8', delimiter=',')

    error_columns = ["ER_MNSE","ER_Mean_Error","General_MNSE","General_Mean_Error","Mean_Error_90","Prim_mdot_Crit_Error[%]","Prim_mdot_Error[%]","Sec_mdot_Crit_Error[%]","Sec_mdot_Error[%]"]

    # Create Cluter Plots
    cluster_plot(error_df.drop(["Momentum", "Learning_Rate", "Iteration"], axis=1))

    # Get first view of error correlation - Important but not definitive
    get_error_correlation(error_df.drop(["Momentum","Learning_Rate","Iteration"], axis=1), error_columns, "Yes")

    nodes_dfs = separate_by(error_df, "Nodes")

    # Get error correlation by nodes number
    for nodes in nodes_dfs:

        get_error_correlation(nodes_dfs[nodes].drop(["Momentum","Learning_Rate","Iteration"], axis=1), error_columns, "Yes", nodes)
        get_Slope_Chart(nodes_dfs[nodes], "Batch_Size", "Dropout_Rate", "Sec_mdot_Crit_Error[%]", nodes)

    # Create Distributed Dot Plot with Nodes
    dot_plot(error_df.drop(["Momentum","Learning_Rate","Iteration"], axis=1))

    # plt.show()

    quit()


if __name__ == '__main__':

    see_all()
    main()
