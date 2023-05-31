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

def script_main():

    # Read the inventory from the correspondent ETROC into a dataframe
    inventory_df = pandas.read_csv(os.path.join(Path(args.etroc), "inventory{}.csv".format(args.etroc)))
                
    # Get all folders in the Parent folder (os.path.dirname(Path))
    runs = [run for run, status in zip(inventory_df['TxtFile'], inventory_df['Status']) if status != '-']
    #runs = [run for run in os.listdir(Path(args.etroc)) if os.path.isdir(os.path.join(Path(args.etroc), run))]
    for run in runs: 
        print("run"), print(run)
        if os.path.isdir(os.path.join(args.etroc, run)):
            print(f'python {args.script} --out-directory {args.etroc}\{run}')
            os.system(f'python {args.script} --out-directory {args.etroc}\{run}')
            print("\n")


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
        '-script',
        '--script',
        help = 'Choose script. Default: "analyse_time_resolution.py"',
        default = "analyse_time_resolution.py",
        dest = 'script',
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
    parser.add_argument(
        '-o',
        '--out-directory',
        metavar = 'path',
        help = 'Path to the output directory for the run data. Default: ./ApplyToEveryRun',
        default = "./ApplyToEveryRun",
        dest = 'out_directory',
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

    script_main()