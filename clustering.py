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
import hdbscan
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

    color_scales = ['greys', 'reds', 'blues','darkmint', 'greens','oranges','purples','purd']
    
    if Matisse.task_completed("calculate_times_in_ns"):
        with Matisse.handle_task("clustering", drop_old_data=drop_old_data) as Miso:
            with sqlite3.connect(Miso.path_directory/"calculate_times_in_ns"/'data.sqlite') as input_sqlite3_connection, \
                 sqlite3.connect(Miso.task_path/'data.sqlite') as output_sqlite3_connection:
                data_df = pandas.read_sql('SELECT * FROM etroc1_data', input_sqlite3_connection, index_col=None)
                
                # SCALE COLUMNS OF INTEREST
                interest = data_df[['calibration_code', 'time_over_threshold', 'time_of_arrival']]
                scaler = StandardScaler()
                scaler.fit(interest)
                data_df['calibration_code_scaled'], data_df['time_over_threshold_scaled'], data_df['time_of_arrival_scaled'] = scaler.transform(interest).T
                interest_variables = ['calibration_code_scaled','time_over_threshold_scaled','time_of_arrival_scaled']

                # OUTPUT: SCATTER WITH DIFFERENT COLORS
                # ORGANIZE EVENTS PER BOARD

                # Cycle over boards
                for board_id in sorted(data_df['data_board_id'].unique()):
                    board_df = data_df[data_df['data_board_id'] == board_id]

                    # KMEANS clustering

                    if method == "KMEANS":

                        k = 8  # Number of clusters
                        n_init_value = 10  # Set the desired value for n_init
                        max_iter = 10 # Max nr of iterations
                        kmeans = KMeans(n_clusters=k, max_iter=max_iter, n_init=n_init_value)
                        kmeans.fit(board_df[interest_variables]) 
                        board_df['Cluster Label'] = kmeans.labels_
                    
                    # DBSCAN clustering

                    if method == "DBSCAN":   

                        min_cluster_size = 5  # Minimum number of samples in a cluster
                        min_samples = 5  # Min number of samples in a neighborhood to form a core point
                        #eps = 0.15 # Max distance between two samples to be considered in the same neighborhood
                        #dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                        dbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
                        dbscan.fit(board_df[interest_variables])

                        board_df['Cluster Label'] = dbscan.labels_

                    # GENERAL

                    unique_labels = sorted(set(board_df['Cluster Label']))
                    print("Board ID:", board_id)
                    print("Cluster Labels:", unique_labels)

                    # ALL CLUSTERS PLOT
                    
                    fig = go.Figure()
                    # Iterate over clusters
                    for cluster_label in unique_labels:
                        cluster = board_df[board_df['Cluster Label'] == cluster_label]        
                        fig.add_trace(go.Scatter(
                            x=cluster['time_over_threshold'],
                            y=cluster['time_of_arrival'],
                            mode='markers',
                            name=f'Cluster {cluster_label}',
                            marker=dict(
                                size=6,
                                color=cluster.index,  # Differentiate points within a cluster using index
                                colorscale=color_scales[cluster_label]  # Color scale for points within a cluster
                            )
                        ))

                    # After the cycle over all clusters
                    fig.update_layout(
                        title=f"Scatter Plot of TOT vs TOA (Board {board_id})",
                        xaxis_title="Time over Threshold",
                        yaxis_title="Time of Arrival"
                    )

                    fig.write_html(
                        Miso.task_path / f'Board{board_id}_TOT_vs_TOA_{args.method}_Cluster_All.html',
                        full_html=False,
                        include_plotlyjs='cdn'
                    )

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
        
        # if Oberon.task_completed("clustering") and make_plots:
        #     plot_etroc1_task(Oberon, "plot_after_clustering", Oberon.path_directory/"data"/"data.sqlite")
        # else:
        #     print("Task clustering was not completed")


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