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
import os
import subprocess
import multiprocessing

# cd OneDrive - Universidade de Lisboa\PIC\ETROC\Scripts
# python all.py --script cut_etroc1_single_run.py --time-cuts time_cuts-hv190.csv --cluster {args.cluster}

def run_analysis(run, index, args):

    print(f"Run: {run}")

    commands_cuts = [
        #f'python process_etroc1_single_run_txt.py --out-directory {args.etroc}/{run} --file {args.file} --time-cuts {args.time_cuts_file} --etroc-number {args.etroc} --method {args.method} --scaling-order {args.sorder} --scaling-method {args.smethod} --log-level {args.log_level} --max_toa {args.max_toa} --max_tot {args.max_tot} --cluster {args.cluster}',
        f'python cut_etroc1_single_run.py --out-directory {args.etroc}/{run} --file {args.file} --time-cuts {args.time_cuts_file} --etroc-number {args.etroc} --method {args.method} --scaling-order {args.sorder} --scaling-method {args.smethod} --log-level {args.log_level} --max_toa {args.max_toa} --max_tot {args.max_tot} --cluster {args.cluster}',
        f'python calculate_times_in_ns.py --out-directory {args.etroc}/{run} --file {args.file} --time-cuts {args.time_cuts_file} --etroc-number {args.etroc} --method {args.method} --scaling-order {args.sorder} --scaling-method {args.smethod} --log-level {args.log_level} --max_toa {args.max_toa} --max_tot {args.max_tot} --cluster {args.cluster}',
        f'python cut_times_in_ns.py --out-directory {args.etroc}/{run} --file {args.file} --time-cuts {args.time_cuts_file} --etroc-number {args.etroc} --method {args.method} --scaling-order {args.sorder} --scaling-method {args.smethod} --log-level {args.log_level} --max_toa {args.max_toa} --max_tot {args.max_tot} --cluster {args.cluster}',
        f'python calculate_time_walk_correction.py --out-directory {args.etroc}/{run} --file {args.file} --time-cuts {args.time_cuts_file} --etroc-number {args.etroc} --method {args.method} --scaling-order {args.sorder} --scaling-method {args.smethod} --log-level {args.log_level} --max_toa {args.max_toa} --max_tot {args.max_tot} --cluster {args.cluster}',
        f'python analyse_time_resolution.py --out-directory {args.etroc}/{run} --file {args.file} --time-cuts {args.time_cuts_file} --etroc-number {args.etroc} --method {args.method} --scaling-order {args.sorder} --scaling-method {args.smethod} --log-level {args.log_level} --max_toa {args.max_toa} --max_tot {args.max_tot} --cluster {args.cluster}',
    ]

    commands_clustering = [
        f'python calculate_times_in_ns.py --out-directory {args.etroc}/{run} --file {args.file} --time-cuts {args.time_cuts_file} --etroc-number {args.etroc} --method {args.method} --scaling-order {args.sorder} --scaling-method {args.smethod} --log-level {args.log_level} --max_toa {args.max_toa} --max_tot {args.max_tot} --cluster {args.cluster}',
        f'python calculate_time_walk_correction.py --out-directory {args.etroc}/{run} --file {args.file} --time-cuts {args.time_cuts_file} --etroc-number {args.etroc} --method {args.method} --scaling-order {args.sorder} --scaling-method {args.smethod} --log-level {args.log_level} --max_toa {args.max_toa} --max_tot {args.max_tot} --cluster {args.cluster}',
        f'python analyse_time_resolution.py --out-directory {args.etroc}/{run} --file {args.file} --time-cuts {args.time_cuts_file} --etroc-number {args.etroc} --method {args.method} --scaling-order {args.sorder} --scaling-method {args.smethod} --log-level {args.log_level} --max_toa {args.max_toa} --max_tot {args.max_tot} --cluster {args.cluster}',
    ]

    if args.script == "clustering.py":
        commands = [f'python {args.script} --out-directory {args.etroc}/{run} --file {args.file} --time-cuts {args.time_cuts_file} --etroc-number {args.etroc} --method {args.method} --scaling-order {args.sorder} --scaling-method {args.smethod} --log-level {args.log_level} --max_toa {args.max_toa} --max_tot {args.max_tot} --cluster {args.cluster}']
    elif args.cluster != "NA":
        commands = commands_clustering
    elif args.time_cuts_file != "time_cuts.csv":
        commands = commands_cuts
    else:
        print("You need to give me '--cluster number' or '--time-cuts time_cuts-hv.csv'.")
        exit()

    # Find the index of the matching script in the commands list (finds the second word/element)
    start_index = next((index for index, command in enumerate(commands) if args.script.startswith(command.split()[1])), None)

    if start_index is not None:
        for command in commands[start_index:]:
            print(f"Start ({index}/{22}): {command.split()[1]}")
            # Execute each command with the appropriate run and arguments
            this_command = f'{command} --out-directory {args.etroc}/{run} --file {args.file} --time-cuts {args.time_cuts_file} --etroc-number {args.etroc} --method {args.method} --scaling-order {args.sorder} --scaling-method {args.smethod} --log-level {args.log_level} --max_toa {args.max_toa} --max_tot {args.max_tot}'
            try:
                # This is where it is happening
                
                process = subprocess.Popen(this_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()

                if process.returncode != 0:
                    error_message = stderr.decode() if stderr is not None else "Unknown error"
                    print(f"Error occurred while running the command for {run}: {error_message}")
                
                print(f"Done ({index+1}/{22}): {command.split()[1]} on {run}")
                
            except Exception as e:
                print(f"Error occurred while running the command for {run}: {str(e)}")
    else:
        print(f"No matching command found for {args.script}")
    
def script_main(args):

    # Read the inventory from the correspondent ETROC into a dataframe
    inventory_df = pandas.read_csv(os.path.join(Path(args.etroc), "inventory{}.csv".format(args.etroc))) 
    # Get all folders in the Parent folder (os.path.dirname(Path))
    runs = [run for run, status in zip(inventory_df['TxtFile'], inventory_df['Status']) if status != '-']
    
    # Using Multiprocessing
    pool = multiprocessing.Pool(processes=5)  # Set the number of processes to 6
    pool.starmap(run_analysis, [(run, index+1, args) for index, run in enumerate(runs)])
    pool.close()
    pool.join()
    
    if args.script != "clustering.py":
        # Analyse time resolution vs bias voltage
        print("\nTime to plot Time Resolution vs Bias Voltage")
        try:
            subprocess.run(f'python analyse_time_resolution_vs_bias_voltage.py --time-cuts {args.time_cuts_file} --cluster {args.cluster}', shell=True)
        except subprocess.CalledProcessError as e:
            # Handle the case when the subprocess call returns a non-zero exit code
            print(f"Failed with exit code {e.returncode}: {e.output}")
            # Add your desired error handling logic here

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ETROC 1 Analysis')
    #################### Needed for: process_etroc1_single_run
    parser.add_argument(
        '--file',
        metavar = 'path',
        help = 'Path to the txt file with the measurements.',
        dest = 'file',
        default = "ETROC1\original\F5P5_F17P5_B2P5_Beam_HV190.txt",
        type = str,
    )
    #################### Needed: ALWAYS
    parser.add_argument(
        '-script',
        '--script',
        help = 'Choose script. Default: "analyse_time_resolution.py"',
        default = "analyse_time_resolution.py",
        required = True,
        dest = 'script',
        type = str,
    )
    #################### Needed for: cut_times_in_ns
    parser.add_argument(
        '-time-cuts',
        '--time-cuts',
        help = 'Selected time cuts csv. Default: "time_cuts.csv"',
        dest = 'time_cuts_file',
        default = "time_cuts.csv",
        #required = True,
        type = str,
    )
    #################### Needed for: calculate_times_in_ns, calculate_time_walk_correction, analyse
    parser.add_argument(
        '-c',
        '--cluster',
        metavar = 'int',
        help = 'Number of the cluster to be selected. Default: "NA"',
        default = "NA",
        dest = 'cluster',
        type = str,
    )
    ####################
    ####################
    ####################
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
        '-etroc',
        '--etroc-number',
        help = 'Path to the ETROC correspondent to the data. Default: ETROC1',
        default = "ETROC1",
        dest = 'etroc',
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
        '-k',
        '--keep-events',
        help = 'Normally, when applying cuts if a certain board does not have data for a given event, the cut will remove that event. If set, these events will be kept instead.',
        action = 'store_true',
        dest = 'keep_events_without_data',
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

    script_main(args)