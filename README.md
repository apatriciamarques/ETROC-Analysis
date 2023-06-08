# ETROC Analysis

Based on https://github.com/cbeiraod/ETROC-Analysis-Scripts.git

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


\begin{enumerate}
    \item \textbf{process\_etroc1\_single\_run\_txt.py (data injection)}
        \begin{enumerate}
            \item process\_etroc1\_data\_run\_txt (original data file)
            \item data (data.sqlite)
            \item plot\_before\_cuts (\textbf{Initial:} histograms, heat maps, scatter plots)
        \end{enumerate}
    \item cut\_etroc1\_single\_run.py (cuts definition: segmentations, data phases)  
            \begin{enumerate}
            \item apply\_event\_cuts (\textbf{Each event cut:} histograms, heat maps, scatter plots)
            \item plot\_after\_cuts (\textbf{Result event cuts:} histograms, heat maps, scatter plots)
        \end{enumerate}
    \item \textbf{calculate\_times\_in\_ns.py (data transformation)}
        \begin{enumerate}
            \item calculate\_times\_in\_ns (data.sqlite)
            \item plot\_times\_in\_ns\_before\_cuts (\textbf{Initial ns:} histograms, heat maps, scatter plots, correlation matrices)
            \item plot\_times\_in\_ns\_after\_cuts (\textbf{Result event cuts [ns]:} histograms, heat maps, scatter plots, correlation matrices)
        \end{enumerate}
    \item cut\_times\_in\_ns.py (time cuts definition: segmentations, data phases)  
        \begin{enumerate}
            \item apply\_time\_cuts (\textbf{Each time cut:} histograms, heat maps, scatter plots)
            \item plot\_after\_time\_cuts (\textbf{Result time cuts:} histograms, heat map, scatter plots)
            \item plot\_time\_after\_time\_cuts (\textbf{Result time cuts [ns]:} histograms, heat maps, scatter plots, correlation matrices)
        \end{enumerate}
    \item \textbf{calculate\_time\_walk\_correction.py (data processing)}
        \begin{enumerate}
            \item calculate\_time\_walk\_correction (\textbf{Each Iteration:} heat maps with parameters, scatter plots, correlation matrices)
        \end{enumerate}
    \item analyse\_time\_resolution.py
        \begin{enumerate}
            \item analyse\_time\_resolution (\textbf{Each Cut and Iteration:} time walk corrections [histograms time diff and time delta] and plots [time resolution vs iteration or cut])
        \end{enumerate}
\end{enumerate}

New Scripts

1. apply\_script\every\_run.py
   - allows to select a script and perform it for all runs
2. clustering.py
   - performs a clustering algorithm in the original data
3. analyse\_time\_resolution\_vs\_bias\_voltage.py
   - creates a simple plot of final TR vs middle board voltage

Note: analyse\_dac\_vs\_charge.py is only done after "charge injection".

Here are the terminal instructions:

```bash
# Activate virtual environment
cd "OneDrive - Universidade de Lisboa\PIC\ETROC-Analysis-Scripts-main\venv\Scripts"
activate.bat

# Run scripts
cd "OneDrive - Universidade de Lisboa\PIC\ETROC\Scripts"
python [process_etroc1_single_run_txt].py [--file original\F5P5_F17P5_B2P5_Beam_HV225.txt] --out-directory [./]ETROC1\F5P5_F17P5_B2P5_Beam_HV225
```
