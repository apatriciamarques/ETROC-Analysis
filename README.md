# ETROC Analysis

Based on https://github.com/cbeiraod/ETROC-Analysis-Scripts.

There's the option to perform data selection either with cutting or clustering.

We can choose to process each data file individually or all at once with a chosen method.

# Individually with cutting

### Process each data file individually

1. **process\_etroc1\_single\_run\_txt.py (data injection)**
   - process\_etroc1\_data\_run\_txt (original data file)
   - data (data.sqlite)
   - plot\_before\_cuts (Initial: histograms, heat maps, scatter plots)

2. cut\_etroc1\_single\_run.py (cuts definition: segmentations, data phases)
   - apply\_event\_cuts (Each event cut: histograms, heat maps, scatter plots)
   - plot\_after\_cuts (Result event cuts: histograms, heat maps, scatter plots)

3. **calculate\_times\_in\_ns.py (data transformation)**
   - calculate\_times\_in\_ns (data.sqlite)
   - plot\_times\_in\_ns\_before\_cuts (Initial ns: histograms, heat maps, scatter plots, correlation matrices)
   - plot\_times\_in\_ns\_after\_cuts (Result event cuts [ns]: histograms, heat maps, scatter plots, correlation matrices)

4. cut\_times\_in\_ns.py (time cuts definition: segmentations, data phases)
   - apply\_event\_cuts (Each time cut: histograms, heat maps, scatter plots)
   - plot\_after\_time\_cuts (Result time cuts: histograms, heat map, scatter plots)
   - plot\_time\_after\_time\_cuts (Result time cuts [ns]: histograms, heat maps, scatter plots, correlation matrices)

5. **calculate\_time\_walk\_correction.py (data processing)**
   - calculate\_time\_walk\_correction (Each Iteration: heat maps with parameters, scatter plots, correlation matrices)

6. analyse\_time\_resolution.py
   - analyse\_time\_resolution (Each Cut and Iteration: time walk corrections [histograms time diff and time delta] and plots [time resolution vs iteration or cut])

# Individually with clustering

1. **process\_etroc1\_single\_run\_txt.py (data injection)**

2. **clustering.py**

3. **calculate\_times\_in\_ns.py (data transformation)**

4. **calculate\_time\_walk\_correction.py (data processing)**

5. analyse\_time\_resolution.py

# Process all runs with a chosen method

1. all.py

2. analyse\_time\_resolution\_vs\_bias\_voltage.py

# Terminal instructions

```bash
# Activate virtual environment
cd {working_environment}\venv\Scripts
activate.bat

# Run scripts
cd {working_path}
python {script}.py --out-directory {etroc/run} --file {input_data} --time-cuts {time_cuts_file} --etroc-number {etroc} --method {method} --scaling-order {sorder} --scaling-method {smethod} --log-level {log_level} --max_toa {max_toa} --max_tot {max_tot} --cluster {selected_cluster}',
        
```

### Examples:

```bash
# Activate virtual environment
cd OneDrive - Universidade de Lisboa\PIC\ETROC-Analysis-Scripts-main\venv\Scripts
activate.bat

# Run scripts
cd OneDrive - Universidade de Lisboa\PIC\ETROC\Scripts
python cut_times_in_ns.py --time-cuts time_cuts-hv220.csv --out-directory ETROC1\F5P5_F17P5_B2P5_Beam_HV235
        
```
