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
import shutil
import pandas
import numpy as np
import sqlite3

from utilities import plot_etroc1_task
from utilities import build_plots
from utilities import apply_event_filter


def apply_numeric_comparison_to_column(
    column: pandas.Series,
    cut_direction: str,
    value: str,
    callee_info: str,
    ):
    if cut_direction == "<":
        return (column < value)
    elif cut_direction == "<=":
        return (column <= value)
    elif cut_direction == ">":
        return (column > value)
    elif cut_direction == ">=":
        return (column >= value)
    elif cut_direction == "==":
        return (column == value)
    elif cut_direction == "<>":
        return (column != value)
    else:
        raise RuntimeError("Unknown cut direction for {}: {}".format(callee_info, cut_direction))

def data_df_apply_single_cut(
    data_df: pandas.DataFrame,
    board_id:int,
    variable:str,
    cut_type:str,
    cut_value:str,
    keep_nan:bool=False,
    ):
    if keep_nan:
        extra_rows_to_keep = data_df[(variable, board_id)].isna()
    else:
        extra_rows_to_keep = False

    return (apply_numeric_comparison_to_column(data_df[(variable, board_id)], cut_type, cut_value, "single cut") | extra_rows_to_keep)

def df_apply_cut(
    df: pandas.DataFrame,
    data_df: pandas.DataFrame,
    board_id:str,
    variable:str,
    cut_type:str,
    cut_value:str,
    keep_nan:bool=False,
    ):
    if board_id != "*" and board_id != "#":
        df['accepted'] &= data_df_apply_single_cut(data_df, int(board_id), variable, cut_type, cut_value, keep_nan=keep_nan)
    else:
        full_cut = None
        board_ids = data_df.stack().reset_index(level="data_board_id")["data_board_id"].unique()
        for this_board_id in board_ids:
            cut = data_df_apply_single_cut(data_df, int(this_board_id), variable, cut_type, cut_value, keep_nan=keep_nan)
            if full_cut is None:
                full_cut = cut
            else:
                if board_id == "*":
                    full_cut &= cut
                elif board_id == "#":
                    full_cut |= cut
                else:  # WTF
                    raise RuntimeError("WTF is going on...")
        df['accepted'] &= full_cut

    return df

def apply_event_cuts(
    data_df: pandas.DataFrame,
    cuts_df: pandas.DataFrame,
    script_logger: logging.Logger,
    Johnny: RM.TaskManager,
    keep_events_without_data:bool = False,
    ):
    """
    Given a dataframe `cuts_df` with one cut per row, e.g.
    ```
               variable  board_id  cut_type  cut_value
       calibration_code         1         <        200
       calibration_code         0         >        140
    time_over_threshold         3        >=        300
    ```
    this function returns a series with the index `event` and the value
    either `True` or `False` stating if the even satisfies ALL the
    cuts at the same time.
    """
    board_id_list = data_df['data_board_id'].unique()
    print("here")
    for board_id in cuts_df['board_id'].unique():
        if board_id != "*" and board_id != "#":
            if int(board_id) not in board_id_list:
                raise ValueError("The board_id defined in the cuts file ({}) can not be found in the data. The set of board_ids defined in data is: {}".format(board_id, board_id_list))

    pivot_data_df = data_df.pivot(
        index = 'event',
        columns = 'data_board_id',
        values = list(set(data_df.columns) - {'data_board_id', 'event'}),
    )

    base_path = Johnny.task_path.resolve()/"CutflowPlots"
    base_path.mkdir(exist_ok=True)

    triggers_accepted_df = pandas.DataFrame({'accepted': True}, index=pivot_data_df.index)
    for idx, cut_row in cuts_df.iterrows():
        triggers_accepted_df = df_apply_cut(triggers_accepted_df, pivot_data_df, cut_row['board_id'], cut_row['variable'], cut_row['cut_type'], cut_row['cut_value'], keep_nan=keep_events_without_data)

        if "output" in cut_row and isinstance(cut_row["output"], str):
            script_logger.info("Making partial cut plots after cut {}:\n{}".format(idx, cut_row))
            base_name = str(idx) + "-" + cut_row["output"]
            (base_path/base_name).mkdir(exist_ok=True)
            this_data_df = apply_event_filter(data_df, triggers_accepted_df)
            #build_plots(this_data_df, Johnny.run_name, Johnny.task_name, base_path/base_name, extra_title="Partial Cuts")
            del this_data_df

    return triggers_accepted_df

def apply_event_cuts_task(
    AdaLovelace: RM.RunManager,
    script_logger: logging.Logger,
    drop_old_data:bool=True,
    keep_events_without_data:bool = False,
):
    if AdaLovelace.task_completed("process_etroc1_data_run") or AdaLovelace.task_completed("process_etroc1_data_run_txt"):
        with AdaLovelace.handle_task("apply_event_cuts", drop_old_data=drop_old_data) as Miso:
            
            if not (Miso.path_directory/"cuts.csv").is_file():
                script_logger.info("A cuts file is not defined for run {}".format(AdaLovelace.run_name))
                print("Attention: a cuts file is not defined.")
            else:
                with sqlite3.connect(Miso.path_directory/"data"/'data.sqlite') as input_sqlite3_connection:
                    cuts_df = pandas.read_csv(Miso.path_directory/"cuts.csv")

                    if ("board_id" not in cuts_df or
                        "variable" not in cuts_df or
                        "cut_type" not in cuts_df or
                        "cut_value" not in cuts_df
                        ):
                        script_logger.error("The cuts file does not have the correct format")
                        raise RuntimeError("Bad cuts config file")
                    cuts_df.to_csv(Miso.task_path/'cuts.backup.csv', index=False)

                    input_df = pandas.read_sql('SELECT * FROM etroc1_data', input_sqlite3_connection, index_col=None)

                    filtered_events_df = apply_event_cuts(input_df, cuts_df, script_logger=script_logger, Johnny=Miso, keep_events_without_data=keep_events_without_data)

                    script_logger.info('Saving run event filter metadata...')
                    filtered_events_df.reset_index().to_feather(Miso.task_path/'event_filter.fd')
                    filtered_events_df.reset_index().to_feather(Miso.path_directory/'event_filter.fd')
                    print(f"I've saved 'event_filter.fd' to {Miso.task_path} and {Miso.path_directory}")
    else:
        print("Task process_etroc1_data_run not completed")

def script_main(
        output_directory:Path,
        drop_old_data:bool=True,
        make_plots:bool=True,
        keep_events_without_data:bool=False,
        ):

    script_logger = logging.getLogger('apply_event_cuts')

    with RM.RunManager(output_directory.resolve()) as Bob:
        Bob.create_run(raise_error=False)

        apply_event_cuts_task(
            Bob,
            script_logger=script_logger,
            drop_old_data=drop_old_data,
            keep_events_without_data=keep_events_without_data,
        )

        if Bob.task_completed("apply_event_cuts") and make_plots:
            #def plot_etroc1_task(Bob_Manager:RM.RunManager,task_name:str,data_file:Path,filter_files:dict[str,Path] = {}, drop_old_data:bool = True,extra_title: str = "",):
            plot_etroc1_task(Bob, "plot_after_cuts", Bob.path_directory/"data"/"data.sqlite", filter_files={"event": Bob.path_directory/"event_filter.fd"})
        else:
            print("Task apply_event_cuts was not completed")



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
        type = str,
    )
    parser.add_argument(
        '-k',
        '--keep-events',
        help = 'Normally, when applying cuts if a certain board does not have data for a given event, the cut will remove that event. If set, these events will be kept instead.',
        action = 'store_true',
        dest = 'keep_events_without_data',
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
        '-c',
        '--cluster',
        metavar = 'int',
        help = 'Number of the cluster to be selected. Default: "NA"',
        default = "NA",
        dest = 'cluster',
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

    script_main(Path(args.out_directory), keep_events_without_data=args.keep_events_without_data)
    
