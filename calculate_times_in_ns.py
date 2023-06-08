#############################################################################
# zlib License
#
# (C) 2023 Cristóvão Beirão da Cruz e Silva <cbeiraod@cern.ch>
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
#
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
#
# 1. The origin of this software must not be misrepresented; you must not
#    claim that you wrote the original software. If you use this software
#    in a product, an acknowledgment in the product documentation would be
#    appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
#    misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.
#############################################################################

from pathlib import Path # Pathlib documentation, very useful if unfamiliar:
                         #   https://docs.python.org/3/library/pathlib.html

import lip_pps_run_manager as RM

import logging
import pandas
import numpy
import sqlite3

from utilities import plot_times_in_ns_task


def calculate_times_in_ns_task(
    Fermat: RM.RunManager,
    script_logger: logging.Logger,
    drop_old_data:bool=True,
    fbin_choice:str="mean",#"mean",
    ):

    # Patrícia added
    if args.cluster == "NA":
        completed_task = "apply_event_cuts"
    else:
        completed_task = "clustering"

    if Fermat.task_completed(completed_task):
        with Fermat.handle_task("calculate_times_in_ns", drop_old_data=drop_old_data) as Einstein:
            
            # Patrícia added
            if args.cluster == "NA":
                completed_task = "apply_event_cuts"
                connect_task = Einstein.path_directory/"data"
            else:
                completed_task = "clustering"
                connect_task = Einstein.get_task_path("clustering")

            with sqlite3.connect(connect_task/'data.sqlite') as input_sqlite3_connection, \
                 sqlite3.connect(Einstein.task_path/'data.sqlite') as output_sqlite3_connection:
                print("path directory"), print(connect_task)
                data_df = pandas.read_sql('SELECT * FROM etroc1_data', input_sqlite3_connection, index_col=None)

                # Patrícia added
                if args.cluster == "NA":

                    print("I'm filtering the data with event cuts")

                    filter_df = pandas.read_feather(Einstein.path_directory/"event_filter.fd")
                    filter_df.set_index("event", inplace=True)

                    from cut_etroc1_single_run import apply_event_filter
                    data_df = apply_event_filter(data_df, filter_df)

                # Patrícia added
                if args.cluster != "NA":

                    print(f"I'm filtering the data choosing cluster number {args.cluster}")
                    data_df['accepted'] = data_df['Cluster Label'] == int(args.cluster)
                
                accepted_data_df = data_df.loc[data_df['accepted']==True]
                board_grouped_accepted_data_df = accepted_data_df.groupby(['data_board_id'])

                board_info_df = board_grouped_accepted_data_df[['calibration_code']].mean()
                board_info_df.rename(columns = {'calibration_code':'calibration_code_mean'}, inplace = True)
                board_info_df['calibration_code_median'] = board_grouped_accepted_data_df[['calibration_code']].median()
                board_info_df['fbin_mean'] = 3.125/board_info_df['calibration_code_mean']
                board_info_df['fbin_median'] = 3.125/board_info_df['calibration_code_median']

                #accepted_data_df.set_index("data_board_id", inplace=True)
                #accepted_data_df["fbin"] = board_info_df['fbin_mean']
                #accepted_data_df.reset_index(inplace=True)

                #accepted_data_df["time_of_arrival_ns"] = 12.5 - accepted_data_df['time_of_arrival']*accepted_data_df['fbin']
                #accepted_data_df["time_over_threshold_ns"] = (accepted_data_df["time_over_threshold"]*2 - (accepted_data_df["time_over_threshold"]/32.).apply(numpy.floor))*accepted_data_df['fbin']

                data_df.set_index("data_board_id", inplace=True)
                if fbin_choice == "mean":
                    data_df["fbin"] = board_info_df['fbin_mean']
                elif fbin_choice == "median":
                    data_df["fbin"] = board_info_df['fbin_median']
                elif fbin_choice == "event":
                    data_df["fbin"] = 3.125/data_df['calibration_code']
                data_df.reset_index(inplace=True)

                data_df["time_of_arrival_ns"] = 12.5 - data_df['time_of_arrival']*data_df['fbin']
                data_df["time_over_threshold_ns"] = (data_df["time_over_threshold"]*2 - (data_df["time_over_threshold"]/32.).apply(numpy.floor))*data_df['fbin']

                board_info_df.to_sql('board_info_data',
                                     output_sqlite3_connection,
                                     #index=False,
                                     if_exists='replace')

                if args.cluster == "NA":
                    data_df.drop(labels=['accepted', 'event_filter'], axis=1, inplace=True)
                if args.cluster != "NA":
                    data_df.drop(labels=['accepted'], axis=1, inplace=True)
                data_df.to_sql('etroc1_data',
                               output_sqlite3_connection,
                               index=False,
                               if_exists='replace')

def script_main(
    output_directory:Path,
    make_plots:bool=True,
    max_toa:float=0,
    max_tot:float=0,
    ):

    # Patrícia added
    if args.cluster == "NA":
        completed_task = "apply_event_cuts"
    else:
        completed_task = "clustering"
    
    script_logger = logging.getLogger(completed_task)

    if max_toa == 0:
        max_toa = None
    if max_tot == 0:
        max_tot = None

    with RM.RunManager(output_directory.resolve()) as Fermat:
        Fermat.create_run(raise_error=False)

        if not Fermat.task_completed(completed_task):
            raise RuntimeError(f"You can only run this script after {completed_task}")

        calculate_times_in_ns_task(Fermat, script_logger=script_logger)

        if make_plots and args.cluster == "NA":
            plot_times_in_ns_task(
                Fermat,
                script_logger=script_logger,
                task_name="plot_times_in_ns_before_cuts",
                data_file=Fermat.get_task_path("calculate_times_in_ns")/'data.sqlite',
                filter_files={},
                max_toa=max_toa,
                max_tot=max_tot,
                min_toa=0,
                min_tot=0,
            )

            plot_times_in_ns_task(
                Fermat,
                script_logger=script_logger,
                task_name="plot_times_in_ns_after_cuts",
                data_file=Fermat.get_task_path("calculate_times_in_ns")/'data.sqlite',
                filter_files={"event": Fermat.path_directory/"event_filter.fd"},
                max_toa=max_toa,
                max_tot=max_tot,
                min_toa=0,
                min_tot=0,
            )

            print(f"I'm printing plot times in ns for cluster ({args.cluster})")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Converts data taken with the KC 705 FPGA development board connected to an ETROC1 into our data format')
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
        required = True,
        type = str,
    )
    parser.add_argument(
        '-a',
        '--max_toa',
        metavar = 'int',
        help = 'Maximum value of the time of arrival (in ns) for plotting. Default: 0 (automatically calculated)',
        default = 0,
        dest = 'max_toa',
        type = float,
    )
    parser.add_argument(
        '-t',
        '--max_tot',
        metavar = 'int',
        help = 'Maximum value of the time over threshold (in ns) for plotting. Default: 0 (automatically calculated)',
        default = 0,
        dest = 'max_tot',
        type = float,
    )
    parser.add_argument(
        '-etroc',
        '--etroc-number',
        help = 'Path to the ETROC correspondent to the data. Default: ETROC1',
        default = "ETROC1",
        dest = 'etroc',
        type = str,
    )
    parser.add_argument(
        '-time-cuts',
        '--time-cuts',
        help = 'Selected time cuts csv. Default: "time_cuts.csv"',
        dest = 'time_cuts_file',
        default = "time_cuts.csv",
        type = str,
    )
    parser.add_argument(
        '-c',
        '--cluster',
        metavar = 'int',
        help = 'Number of the cluster to be selected. Default: "NA"',
        default = "NA",
        dest = 'cluster',
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
    parser.add_argument(
        '--file',
        metavar = 'path',
        help = 'Path to the txt file with the measurements.',
        #required = True,
        dest = 'file',
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

    script_main(Path(args.out_directory), max_toa=args.max_toa, max_tot=args.max_tot)