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

from utilities import filter_dataframe
from utilities import make_multi_scatter_plot
from utilities import make_time_correlation_plot
from utilities import make_board_scatter_with_fit_plot

import scipy.odr
import plotly.express as px
import plotly.graph_objects as go


def calculate_delta_toa(
    iteration:int,
    board_list:list[int],
    original_df:pandas.DataFrame,
    data_df:pandas.DataFrame,
    pivot_df:pandas.DataFrame,
    ):
    delta_column = "delta_toa_to_reference_iteration_{}".format(iteration)
    if iteration > 0:
        delta_column_twc = "delta_toa_to_reference_iteration_{}_twc".format(iteration - 1)
    for board_id in board_list:
        delta_board_column = (delta_column, board_id)
        if iteration > 0:
            delta_board_column_twc = (delta_column_twc, board_id)
            prev_columns_to_sum = []

        columns_to_sum = []
        for other_board in board_list:
            if other_board == board_id:
                continue
            if iteration == 0:
                columns_to_sum += [("time_of_arrival_ns", other_board)]
            else:
                columns_to_sum += [("time_of_arrival_twc_iteration_{}".format(iteration-1), other_board)]

                if iteration == 1:
                    prev_columns_to_sum += [("time_of_arrival_ns", other_board)]
                else:
                    prev_columns_to_sum += [("time_of_arrival_twc_iteration_{}".format(iteration-2), other_board)]

        if iteration > 0:
            pivot_df[delta_board_column_twc] = pivot_df[prev_columns_to_sum].sum(axis=1)/(len(board_list) - 1) - pivot_df[("time_of_arrival_twc_iteration_{}".format(iteration-1), board_id)]
        pivot_df[delta_board_column] = pivot_df[columns_to_sum].sum(axis=1)/(len(board_list) - 1) - pivot_df[("time_of_arrival_ns", board_id)]
    if iteration > 0:
        data_df[delta_column_twc] = pivot_df.stack()[delta_column_twc]
        original_df[delta_column_twc] = data_df[delta_column_twc]
    data_df[delta_column] = pivot_df.stack()[delta_column]
    original_df[delta_column] = data_df[delta_column]

def fit_poly(
    poly_order:int,
    x_column:pandas.Series,
    y_column:pandas.Series,
    ):
    poly_model = scipy.odr.polynomial(poly_order)

    odr_data = scipy.odr.Data(
        x_column,
        y_column
    )

    output = scipy.odr.ODR(odr_data, poly_model).run()
    poly = numpy.poly1d(output.beta[::-1])

    dict1 = dict( ("p{}".format(idx), output.beta[idx]) for idx in range(poly_order+1))

    return poly, dict1

def fit_and_plot_twc(
    Carl: RM.TaskManager,
    iteration:int,
    board_list:list[int],
    data_df:pandas.DataFrame,
    pivot_df:pandas.DataFrame,
    poly_order:int = 2,
    full_html: bool = False,  # For saving a html containing only a div with the plot
    extra_title: str = ""
    ):
    base_path = Carl.task_path
    iteration_path = base_path/"Iteration_{}".format(iteration)
    iteration_path.mkdir(exist_ok=True)
    if iteration > 0:
        previous_iteration_path = base_path/"Iteration_{}".format(iteration-1)

    if extra_title != "":
        extra_title = "<br>" + extra_title

    columns = ["twc_iteration", "board_id", "twc_applied"]
    for idx in range(poly_order+1):
        columns += ["p{}".format(idx)]

    row_list = []
    delta_column = "delta_toa_to_reference_iteration_{}".format(iteration)
    delta_twc_column = "delta_toa_to_reference_iteration_{}_twc".format(iteration - 1)
    for board_id in board_list:
        accepted = pivot_df[("accepted", board_id)]
        x_column = pivot_df.loc[accepted][("time_over_threshold_ns", board_id)].astype(float)

        if iteration > 0:
            y_column = pivot_df.loc[accepted][(delta_twc_column, board_id)].astype(float)

            # Save fit results for storing in dataframe later
            poly, dict1 = fit_poly(poly_order, x_column, y_column)
            dict1["twc_iteration"] = iteration - 1
            dict1["board_id"] = board_id
            dict1["twc_applied"] = True
            row_list.append(dict1)

            make_board_scatter_with_fit_plot(
                data_df = pivot_df,
                base_path = previous_iteration_path,
                run_name = Carl.run_name,
                board_id = board_id,
                x_axis_col = "time_over_threshold_ns",
                y_axis_col = delta_twc_column,
                x_axis_label = 'Time over Threshold [ns]',
                y_axis_label = 'ΔT TWC [ns]',
                title = "Time Walk Correction - Iteration {}".format(iteration - 1),
                file_name = "TWC_After_Correction",
                poly = poly,
                full_html = full_html,
                extra_title = extra_title,
            )

        y_column = pivot_df.loc[accepted][(delta_column, board_id)].astype(float)

        # Save fit results for storing in dataframe later
        poly, dict1 = fit_poly(poly_order, x_column, y_column)
        dict1["twc_iteration"] = iteration
        dict1["board_id"] = board_id
        dict1["twc_applied"] = False
        row_list.append(dict1)

        make_board_scatter_with_fit_plot(
            data_df = pivot_df,
            base_path = iteration_path,
            run_name = Carl.run_name,
            board_id = board_id,
            x_axis_col = "time_over_threshold_ns",
            y_axis_col = delta_column,
            x_axis_label = 'Time over Threshold [ns]',
            y_axis_label = 'ΔT [ns]',
            title = "Time Walk Correction - Iteration {}".format(iteration),
            file_name = "TWC_Before_Correction",
            poly = poly,
            full_html = full_html,
            extra_title = extra_title,
        )


    toa_var = "time_of_arrival_ns"
    toa_label = "Time of Arrival [ns]"
    # The below doesn't really make much sense
    #if iteration > 0:
    #    toa_var = "time_of_arrival_twc_iteration_{}".format(iteration-1)
    #    toa_label = "Time of Arrival - TWC {} [ns]".format(iteration-1)
    make_multi_scatter_plot(
        data_df=data_df.loc[data_df['accepted']],
        run_name=Carl.run_name,
        task_name=Carl.task_name,
        base_path=iteration_path,
        color_column="data_board_id_cat",
        full_html=full_html,
        extra_title=extra_title,
        additional_dimensions=[toa_var, "time_over_threshold_ns", delta_column],
        additional_labels={
            "time_over_threshold_ns": "Time over Threshold [ns]",
            toa_var: toa_label,
            delta_column: "ΔT [ns]",
        },
        use_base_dimensions=False,
    )

    tmp_pivot_df = pivot_df.copy()
    tmp_pivot_df.columns = ["{}_{}".format(x, y) for x, y in tmp_pivot_df.columns]
    make_time_correlation_plot(
        data_df=tmp_pivot_df.loc[tmp_pivot_df['accepted_{}'.format(board_id)]],
        base_path=iteration_path,
        run_name=Carl.run_name,
        board_ids=board_list,
        full_html=full_html,
        extra_title=extra_title,
        additional_dimensions=[delta_column + "_{}".format(id) for id in board_list],
        additional_labels={
            delta_column + "_{}".format(id): "ΔT {} [ns]".format(id) for id in board_list
        },
    )

    return pandas.DataFrame(row_list, columns=columns)

def apply_twc(
    iteration:int,
    board_list:list[int],
    original_df:pandas.DataFrame,
    data_df:pandas.DataFrame,
    pivot_df:pandas.DataFrame,
    fit_df:pandas.DataFrame,
    ):
    fit_info_df = fit_df.query('twc_applied==False').drop(['twc_applied'], axis=1).set_index(['twc_iteration','board_id'])
    toa_column = "time_of_arrival_ns"
    tot_column = "time_over_threshold_ns"
    twc_column = "time_of_arrival_twc_iteration_{}".format(iteration)
    for board_id in board_list:
        toa_board_column = (toa_column, board_id)
        tot_board_column = (tot_column, board_id)
        twc_board_column = (twc_column, board_id)
        poly = numpy.poly1d(fit_info_df.loc[(iteration,board_id)][::-1])

        pivot_df[twc_board_column] = pivot_df[toa_board_column] + poly(pivot_df[tot_board_column])
    data_df[twc_column] = pivot_df.stack()[twc_column]
    original_df[twc_column] = data_df[twc_column]

def calculate_time_walk_correction_task(
    Homer: RM.RunManager,
    script_logger: logging.Logger,
    drop_old_data:bool=True,
    iterations:int=1,
    poly_order:int=2,
    ):
    # Patrícia added
    if args.cluster == "NA":
        completed_task = "apply_time_cuts"
    else:
        completed_task = "calculate_times_in_ns"

    if Homer.task_completed(completed_task):
        with Homer.handle_task("calculate_time_walk_correction", drop_old_data=drop_old_data) as Carl:
            with sqlite3.connect(Carl.get_task_path("calculate_times_in_ns")/'data.sqlite') as input_sqlite3_connection, \
                 sqlite3.connect(Carl.task_path/'data.sqlite') as output_sqlite3_connection:
                print("path directory"), print(Carl.get_task_path("calculate_times_in_ns"))
                original_df = pandas.read_sql('SELECT * FROM etroc1_data', input_sqlite3_connection, index_col=None)
                board_list = sorted(original_df['data_board_id'].unique())

                # Patrícia added conditions
                if args.cluster == "NA":

                    print("I'm filtering the data with event and time cuts")

                    data_df = filter_dataframe(
                        df=original_df,
                        filter_files={
                            "event": Carl.path_directory/"event_filter.fd",
                            "time": Carl.path_directory/"time_filter.fd",
                        },
                        script_logger=script_logger,
                    )

                if args.cluster != "NA":
                    print(f"I'm filtering the data choosing cluster number {args.cluster}")
                    data_df = original_df.copy()
                    data_df['accepted'] = data_df['Cluster Label'] == int(args.cluster)

                pivot_df = data_df.pivot(
                    index = 'event',
                    columns = 'data_board_id',
                    values = list(set(data_df.columns) - {'data_board_id', 'event'}),
                )
                
                print("data_df"),print(data_df)
                print("pivot_df"),print(pivot_df)

                data_df["data_board_id_cat"] = data_df["data_board_id"].astype(str)
                original_df.set_index(["event", "data_board_id"], inplace=True)
                data_df.set_index(["event", "data_board_id"], inplace=True)

                full_fit_df: pandas.DataFrame = None
                for iteration in range(iterations):
                    calculate_delta_toa(iteration, board_list, original_df, data_df, pivot_df)
                    fit_df = fit_and_plot_twc(Carl, iteration, board_list, data_df, pivot_df, poly_order=poly_order)
                    apply_twc(iteration, board_list, original_df, data_df, pivot_df, fit_df)

                    if full_fit_df is None:
                        full_fit_df = fit_df
                    else:
                        full_fit_df = full_fit_df.append(fit_df)
                calculate_delta_toa(iterations, board_list, original_df, data_df, pivot_df)
                fit_df = fit_and_plot_twc(Carl, iterations, board_list, data_df, pivot_df, poly_order=poly_order)
                #I've put _ because of this error: AttributeError: 'DataFrame' object has no attribute 'append'. Did you mean: '_append'?
                full_fit_df = full_fit_df._append(fit_df).query("twc_iteration < {}".format(iterations)).reset_index(drop=True)

                original_df.reset_index(inplace=True)

                twc_df = pandas.DataFrame([{'max_twc_iterations': iterations}], columns=['max_twc_iterations'])

                original_df.to_sql('etroc1_data',
                               output_sqlite3_connection,
                               index=False,
                               if_exists='replace')
                full_fit_df.to_sql('twc_fit_info',
                               output_sqlite3_connection,
                               index=False,
                               if_exists='replace')
                twc_df.to_sql('twc_info',
                               output_sqlite3_connection,
                               index=False,
                               if_exists='replace')

def script_main(
    output_directory:Path,
    make_plots:bool=True,
    iterations:int=1,
    poly_order:int=2,
    ):

    script_logger = logging.getLogger('calculate_twc')

    with RM.RunManager(output_directory.resolve()) as Homer:
        Homer.create_run(raise_error=False)

        # Patrícia added
        if args.cluster == "NA":
            completed_task = "apply_time_cuts"
        else:
            completed_task = "calculate_times_in_ns"

        if not Homer.task_completed(completed_task):
            raise RuntimeError("You can only run this script after applying  time cuts (--cluster NA)\n or after calculating time in ns (--cluster nr)")

        calculate_time_walk_correction_task(Homer, script_logger=script_logger, iterations=iterations, poly_order=poly_order)

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
        '-i',
        '--iterations',
        metavar = 'int',
        help = 'Number of times to iterate the time walk correction calculation. Default: 1',
        default = 1,
        dest = 'iterations',
        type = int,
    )
    parser.add_argument(
        '-p',
        '--order',
        metavar = 'int',
        help = 'Order of the polynomial to use for fitting and applying the time walk correction. Default: 2',
        default = 2,
        dest = 'order',
        type = int,
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

    script_main(Path(args.out_directory), iterations=args.iterations, poly_order=args.order)