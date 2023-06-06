#############################################################################
#
# Patrícia Marques
#
#############################################################################

from pathlib import Path # Pathlib documentation, very useful if unfamiliar:
                         #   https://docs.python.org/3/library/pathlib.html

import lip_pps_run_manager as RM

import logging
import pandas
import numpy
import sqlite3
#import hdbscan
import os
import re

from utilities import make_histogram_plot
from utilities import make_2d_line_plot
from utilities import make_board_scatter_with_fit_plot
from utilities import plot_etroc1_task
from utilities import build_plots
from utilities import apply_event_filter
from utilities import make_tot_vs_toa_plots

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler


import scipy.odr
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as colors
import matplotlib.pyplot as plt
from math import sqrt

def clustering_task(
    Matisse: RM.RunManager,
    script_logger: logging.Logger,
    drop_old_data:bool=True,
    method: str = "",
    ):

    if Matisse.task_completed("process_etroc1_data_run_txt"):
        with Matisse.handle_task("clustering", drop_old_data=drop_old_data) as Miso:
            with sqlite3.connect(Miso.path_directory/"data"/'data.sqlite') as input_sqlite3_connection, \
                 sqlite3.connect(Miso.task_path/'data.sqlite') as output_sqlite3_connection:
                data_df = pandas.read_sql('SELECT * FROM etroc1_data', input_sqlite3_connection, index_col=None)
                print("data_df"), print(data_df)

                data_df['calibration_code'] = numpy.clip(data_df['calibration_code'],135,155)

                # SCALE VARIABLES OF INTEREST CC, TOT, TOA
                if args.sorder == "before_restructure":
                    factor_cc = 1
                    factor_tot = 1
                    factor_toa = 1
                    cc = f"calibration_code"
                    tot = f"time_over_threshold"
                    toa = f"time_of_arrival"
                    cc_scaled = f"calibration_code_scaled"
                    tot_scaled = f"time_over_threshold_scaled"
                    toa_scaled = f"time_of_arrival_scaled"
                    for var, var_scaled, factor in zip([cc, tot, toa], [cc_scaled, tot_scaled, toa_scaled], [factor_cc, factor_tot, factor_toa]):
                        
                        # standard SCALING
                        if args.smethod == "standard":
                            data_df[var_scaled] = factor * (data_df[var] - data_df[var].mean()) / numpy.std(data_df[var])

                        # ROBUST SCALING
                        if args.smethod == "robust":
                            scaler = RobustScaler().fit(data_df[var].to_numpy().reshape(-1, 1))
                            data_df[var_scaled] = scaler.transform(data_df[var].to_numpy().reshape(-1, 1))

                        # MINMAX SCALING
                        if args.smethod == "minmax":
                            scaler = MinMaxScaler().fit(data_df[var].to_numpy().reshape(-1, 1))
                            data_df[var_scaled] = scaler.transform(data_df[var].to_numpy().reshape(-1, 1))
               
                    #########################

                    fig = go.Figure()
                    for board_id in [0,1,3]:
                        fig.add_trace(go.Histogram(
                            x=data_df.loc[data_df["data_board_id"] == board_id]["calibration_code_scaled"],
                            name='Board {}'.format(board_id), # name used in legend and hover labels
                            opacity=0.5,
                            bingroup=1,
                        ))
                    fig.update_layout(
                        barmode='overlay',
                        title_text="Histogram of Calibration Code<br><sup>Run: </sup>",
                        xaxis_title_text='Calibration Code', # xaxis label
                        yaxis_title_text='Count', # yaxis label
                    )
                    fig.update_yaxes(type="log")
                    fig.update_traces(
                        histnorm="probability"
                    )
                    fig.update_layout(
                        yaxis_title_text='Probability', # yaxis label
                    )
                    fig.write_html(
                        Miso.task_path/'calibration_code_scaled_pdf.html',
                        full_html = False,
                        include_plotlyjs = 'cdn',
                    )

                    fig = go.Figure()
                    for board_id in [0,1,3]:
                        fig.add_trace(go.Histogram(
                            x=data_df.loc[data_df["data_board_id"] == board_id]["time_of_arrival_scaled"],
                            name='Board {}'.format(board_id), # name used in legend and hover labels
                            opacity=0.5,
                            bingroup=1,
                        ))
                    fig.update_layout(
                        barmode='overlay',
                        title_text="Histogram of Time of Arrival<br><sup>Run:</sup>",
                        xaxis_title_text='Time of Arrival', # xaxis label
                        yaxis_title_text='Count', # yaxis label
                    )
                    fig.update_traces(
                        histnorm="probability"
                    )
                    fig.update_layout(
                        yaxis_title_text='Probability', # yaxis label
                    )
                    fig.write_html(
                        Miso.task_path/'time_of_arrival_scaled_pdf.html',
                        full_html = False,
                        include_plotlyjs = 'cdn',
                    )

                    fig = go.Figure()
                    for board_id in sorted(data_df["data_board_id"].unique()):
                        fig.add_trace(go.Histogram(
                            x=data_df.loc[data_df["data_board_id"] == board_id]["time_over_threshold_scaled"],
                            name='Board {}'.format(board_id), # name used in legend and hover labels
                            opacity=0.5,
                            bingroup=1,
                        ))
                    fig.update_layout(
                        barmode='overlay',
                        title_text="Histogram of Time over Threshold<br><sup>Run:</sup>",
                        xaxis_title_text='Time over Threshold', # xaxis label
                        yaxis_title_text='Count', # yaxis label
                    )
                    fig.update_traces(
                        histnorm="probability"
                    )
                    fig.update_layout(
                        yaxis_title_text='Probability', # yaxis label
                    )
                    fig.write_html(
                        Miso.task_path/'time_over_threshold_scaled_pdf.html',
                        full_html = False,
                        include_plotlyjs = 'cdn',
                    )

                ######################
                ######################
                ######################
                ######################

                # RESTRUCTURE DATAFRAME
                
                pivot_df = data_df.pivot(index = 'event', columns = 'data_board_id',)
                pivot_df.columns = [f"{col} {board_id}" for col, board_id
                                    in zip(pivot_df.columns.get_level_values(0),
                                           pivot_df.columns.get_level_values(1))] # new column names
                pivot_df = pivot_df.reset_index()
                print("pivot_df"), print(pivot_df)
                
                # SCALING AFTER RESTRUCTURING

                if args.sorder == "after_restructure":
                    factor_cc = 1
                    factor_tot = 1
                    factor_toa = 1
                    for board_id in [0, 1, 3]:
                        cc = f"calibration_code {board_id}"
                        tot = f"time_over_threshold {board_id}"
                        toa = f"time_of_arrival {board_id}"
                        cc_scaled = f"calibration_code_scaled {board_id}"
                        tot_scaled = f"time_over_threshold_scaled {board_id}"
                        toa_scaled = f"time_of_arrival_scaled {board_id}"
                        for var, var_scaled, factor in zip([cc, tot, toa], [cc_scaled, tot_scaled, toa_scaled], [factor_cc, factor_tot, factor_toa]):
                                                    
                            # standard SCALING
                            if args.smethod == "standard":
                                pivot_df[var_scaled] = factor * (pivot_df[var] - pivot_df[var].mean()) / numpy.std(pivot_df[var])

                            # ROBUST SCALING
                            if args.smethod == "robust":
                                scaler = RobustScaler().fit(pivot_df[var].to_numpy().reshape(-1, 1))
                                pivot_df[var_scaled] = scaler.transform(pivot_df[var].to_numpy().reshape(-1, 1))

                            # MINMAX SCALING
                            if args.smethod == "minmax":
                                scaler = MinMaxScaler().fit(pivot_df[var].to_numpy().reshape(-1, 1))
                                pivot_df[var_scaled] = scaler.transform(pivot_df[var].to_numpy().reshape(-1, 1))
                            
            
                # APPLY CLUSTERING ALGORITHM 

                interest_variables = ['calibration_code_scaled 0','time_over_threshold_scaled 0','time_of_arrival_scaled 0',
                                      'calibration_code_scaled 1','time_over_threshold_scaled 1','time_of_arrival_scaled 1',
                                      'calibration_code_scaled 3','time_over_threshold_scaled 3','time_of_arrival_scaled 3']
                print("interest_variables"), print(pivot_df[interest_variables])

                if method == "KMEANS":

                    k = 8 # Number of clusters
                    max_iter = 500 # Max nr of iterations
                    n_init_value = 50 # Set the desired value for n_init
                    # 10: 2/3 clusters along the S
                    kmeans = KMeans(n_clusters=k, max_iter=max_iter, n_init=n_init_value)
                    #kmeans.fit(pivot_df[interest_variables]) 
                    kmeans.fit(pivot_df[interest_variables])
                    '''
                    pivot_df[['time_over_threshold_scaled 0','time_of_arrival_scaled 0',
                                    'time_over_threshold_scaled 1','time_of_arrival_scaled 1',
                                    'time_over_threshold_scaled 3','time_of_arrival_scaled 3']])
                                    '''
                    pivot_df['Cluster Label'] = kmeans.labels_
                
                # DBSCAN clustering

                if method == "DBSCAN":   

                    min_cluster_size = 5  # Minimum number of samples in a cluster
                    min_samples = 5  # Min number of samples in a neighborhood to form a core point
                    #eps = 0.15 # Max distance between two samples to be considered in the same neighborhood
                    #dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    dbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
                    dbscan.fit(pivot_df[interest_variables])
                    pivot_df['Cluster Label'] = dbscan.labels_

                print("pivot_df with cluster labels"), print(pivot_df)

                # GIVE ORIGINAL DATA_DF THE CORRESPONDENT CLUSTER LABELS
                data_df = data_df.merge(pivot_df[['event', 'Cluster Label']], on='event', how='left')
                print("data_df merged"), print(data_df)

                # COLOR MAP

                colormap = {0 : 'blue',
                            1 : 'green',
                            2 : 'red',
                            3 : 'cyan',
                            4 : 'magenta',
                            5 : 'yellow',
                            6 : 'black',
                            7 : 'khaki',
                   }

                # ITERATE OVER THE CLUSTERS TO GET THE SAME COLOR
                
                fig = {}  
                fig_all = go.Figure()
                cc = "calibration_code"
                tot = "time_over_threshold"
                toa = "time_of_arrival"
                for cluster_label in sorted(set(data_df['Cluster Label'])):

                    cluster = data_df[data_df['Cluster Label'] == cluster_label]
                    cluster_color = colormap[cluster_label]

                    # BUILD FIGURE
                    for board_id in sorted(data_df["data_board_id"].unique()):                    

                        # ALL CLUSTERS PLOT

                        if cluster_label == 0:
                            fig[board_id] = go.Figure()
                            print(f"board_id = {board_id}, type = all")
                        
                        fig[board_id].add_trace(go.Scatter(
                            x=cluster.loc[cluster['data_board_id'].astype(int) == int(board_id), tot],
                            y=cluster.loc[cluster['data_board_id'].astype(int) == int(board_id), toa],
                            mode='markers',
                            name=f'Cluster {cluster_label}',
                            marker=dict(
                                size=4,
                                color=cluster_color,  # Differentiate points within a cluster using index
                            )
                        ))

                        fig[board_id].update_layout(
                            title=f"Scatter Plot of TOT vs TOA (Board {board_id}) {args.out_directory}",
                            xaxis_title="Time over Threshold",
                            yaxis_title="Time of Arrival"
                        )

                        fig[board_id].write_html(
                            Miso.task_path / f'Board{board_id}_TOT_vs_TOA_{args.method}_{args.sorder}_{args.smethod}_Cluster_All.html',
                            full_html=False,
                            include_plotlyjs='cdn'
                        )

                    fig_all.add_trace(go.Scatter(
                        x=cluster[tot],
                        y=cluster[toa],
                        mode='markers',
                        name=f'Cluster {cluster_label}',
                        marker=dict(
                            size=4,
                            color=cluster_color,  # Differentiate points within a cluster using index
                        )
                    ))

                    fig_all.update_layout(
                        title=f"Scatter Plot of TOT vs TOA {args.out_directory}",
                        xaxis_title="Time over Threshold",
                        yaxis_title="Time of Arrival"
                    )

                    fig_all.write_html(
                        Miso.task_path / f'TOT_vs_TOA_{args.method}_{args.sorder}_{args.smethod}_Cluster_All.html',
                        full_html=False,
                        include_plotlyjs='cdn'
                    )

                data_df.to_sql('etroc1_data',
                               output_sqlite3_connection,
                               index=False,
                               if_exists='replace')
  
def script_main(
    output_directory:Path,
    drop_old_data:bool=True,
    make_plots:bool=True,
    ):

    script_logger = logging.getLogger('clustering')

    with RM.RunManager(output_directory.resolve()) as Oberon:
        Oberon.create_run(raise_error=False)

        clustering_task(
            Oberon,
            script_logger=logging.Logger,
            method = args.method,
            )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Clustering Method Kmeans')
    parser.add_argument(
        '-l',
        '--log-level',
        help = 'Set the logging level. Default: WARNING',
        choices = ["CRITICAL","ERROR","WARNING","INFO","DEBUG","NOTSET"],
        default = "WARNING",
        dest = 'log_level',
    )
    parser.add_argument(
        '--log-file',
        help = 'If set, the full log will be saved to a file (i.e. the log level is ignored)',
        action = 'store_true',
        dest = 'log_file',
    )
    parser.add_argument(
        '-o',
        '--out-directory',
        metavar = 'path',
        help = 'Path to the output directory for the run data. Default: ./out',
        default = "./out",
        dest = 'out_directory',
        type = str,
    )
    parser.add_argument(
        '-m',
        '--method',
        help = 'Clustering method: "KMEANS" or "DBSCAN". Default: "KMEANS"',
        default = "KMEANS",
        dest = 'method',
        type = str,
    )
    parser.add_argument(
        '-scaling-order',
        '--scaling-order',
        help = 'Scaling before of after restructuring: after_restructure/before_restructure. Default: "before_restructure"',
        default = "before_restructure",
        dest = 'sorder',
        type = str,
    )
    parser.add_argument(
        '-scaling-method',
        '--scaling-method',
        help = 'Scaling method for K Means: standard/minmax/robust. Default: "robust"',
        default = "robust",
        dest = 'smethod',
        type = str,
    )

    args = parser.parse_args()

    if args.log_file:
        logging.basicConfig(filename='logging.log', filemode='w', encoding='utf-8', level=logging.NOTSET)
    else:
        if args.log_level == "CRITICAL":
            logging.basicConfig(level=50)
        elif args.log_level == "ERROR":
            logging.basicConfig(level=40)
        elif args.log_level == "WARNING":
            logging.basicConfig(level=30)
        elif args.log_level == "INFO":
            logging.basicConfig(level=20)
        elif args.log_level == "DEBUG":
            logging.basicConfig(level=10)
        elif args.log_level == "NOTSET":
            logging.basicConfig(level=0)

    script_main(Path(args.out_directory))#os.path.dirname