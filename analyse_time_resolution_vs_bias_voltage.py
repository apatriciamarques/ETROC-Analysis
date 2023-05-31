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
import os
import re

from utilities import filter_dataframe
from utilities import make_histogram_plot
from utilities import make_2d_line_plot
from utilities import make_board_scatter_with_fit_plot

import scipy.odr
import plotly.express as px
import plotly.graph_objects as go
from math import sqrt

def get_voltage(
    Matisse: RM.RunManager,
    script_logger: logging.Logger,
    ):
   
    with Matisse.handle_task("get_voltage") as Ana:
        with sqlite3.connect(Ana.task_path/'data.sqlite') as output_sqlite3_connection:
            data_frames = []
            # Read the inventory from the correspondent ETROC into a dataframe
            inventory_df = pandas.read_csv(os.path.join(Path(args.etroc), "inventory{}.csv".format(args.etroc)))
                        
            # Get all folders in the Parent folder (os.path.dirname(Path))
            runs = [run for run, status in zip(inventory_df['TxtFile'], inventory_df['Status']) if status != '-']
            #runs = [run for run in os.listdir(Path(args.etroc)) if os.path.isdir(os.path.join(Path(args.etroc), run))]
            for run in runs: 
                # From each run, extract the Middle Board applied Voltage plus I2C Configuration
                voltage = float(inventory_df.loc[inventory_df['TxtFile'] == run, 'Voltage Middle (V)'].values[0])
                config = inventory_df.loc[inventory_df['TxtFile'] == run, 'I2C Configuration'].values[0]
                # Get Time Resolution
                analyse_run = os.path.join(Path(args.etroc), run, 'analyse_time_resolution')
                sqlite_file = os.path.join(analyse_run, 'time_resolution.sqlite')
                
                if os.path.isdir(analyse_run) and os.path.isfile(sqlite_file):

                    print("run"), print(run)
                    print("analyse_run"), print(analyse_run)
                    print("sqlite_file"), print(sqlite_file)
                    print("voltage"), print(voltage)
                    print("I2C config"), print(config),print("\n")

                    with sqlite3.connect(sqlite_file) as connection:
                        print("CONNECT"), print("run"), print(run)
                        data_df = pandas.read_sql('SELECT * FROM timing_info', connection, index_col=None)
                    # Filter rows with 'Final' in them
                    data_df = data_df[data_df["step_name"].str.contains('Final')]
                    data_df = data_df[['data_board_id', 'time_resolution_new', 'time_resolution_new_unc']]
                    # Convert times from ns into ps
                    data_df["time_resolution_new"] = data_df["time_resolution_new"]*1000
                    data_df["time_resolution_new_unc"] = data_df["time_resolution_new_unc"]*1000
                    data_df['voltage'] = voltage
                    data_df['I2C Configuration'] = config
                    data_frames.append(data_df)
            
            # Concatenate the list of DataFrames into a single 
            data_frames = pandas.concat(data_frames, ignore_index=True).sort_values(['I2C Configuration','voltage'])
            print(data_frames)
            return data_frames

def plot_time_resolution_vs_bias_voltage_task(
    Oberon: RM.RunManager,
    script_logger: logging.Logger,
    ):

    from math import ceil
    from math import floor

    with Oberon.handle_task("plot_time_resolution_vs_bias_voltage") as Matisse:
        with sqlite3.connect(Matisse.task_path/'time_resolution.sqlite') as output_sqlite3_connection:

            data_df = get_voltage(
                Matisse,
                script_logger=script_logger,
            )

            #SQL
            data_df.to_sql('time_resolution_vs_bias_voltage_data',
                            output_sqlite3_connection,
                            #index=False,
                            if_exists='replace')
            
            make_2d_line_plot(
                    data_df=data_df,
                    run_name=Oberon.run_name,
                    task_name=Matisse.task_name,
                    base_path=Matisse.task_path,
                    plot_title="Time Resolution vs Bias Voltage",
                    subtitle="Board 1 used as trigger and others' applied voltage constant at 230V",
                    x_var=data_df['voltage'],
                    y_var=data_df['time_resolution_new'],
                    y_error=data_df['time_resolution_new_unc'],
                    file_name="time_resolution_vs_bias_voltage",
                    color_var="data_board_id",
                    line_var = 'I2C Configuration',
                    symbol_var = 'I2C Configuration',
                    labels={
                        'I2C Configuration': 'Configuration',
                        'data_board_id': 'Board ID',
                        'time_resolution_new': 'Time Resolution [ps]',
                        'voltage': 'Bias Voltage Board 1 [V]',
                    },
                )


def script_main(
    output_directory:Path,
    ):

    script_logger = logging.getLogger('time_resolution_vs_bias_voltage')

    with RM.RunManager(output_directory.resolve()) as Oberon:
        Oberon.create_run(raise_error=False)

        plot_time_resolution_vs_bias_voltage_task(
            Oberon,
            script_logger=logging.Logger,
            )

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
        help = 'Path to the output directory for the run data. Default: ./BiasVoltage',
        default = "ETROC1/BiasVoltage",#"./BiasVoltage",
        dest = 'out_directory',
        type = str,
    )
    parser.add_argument(
        '-etroc',
        '--etroc-number',
        help = 'Path to the ETROC correspondent to the data. Default: ETROC1',
        default = "ETROC1",
        dest = 'etroc',
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

    script_main(
        Path(args.out_directory),
    )