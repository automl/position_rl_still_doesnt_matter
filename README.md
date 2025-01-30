# Position: Deep RL Still Does Not Matter (Most of The Time)

In our position paper, we use insights from RL papers of recent years to support our arguments. 
This repository contains the code for scraping, processing and plotting these insights:

    .
    ├── ml_papers                           # Download folder for ML papers. Also contains processed files per conference.
    ├── plots                               # All plots
    ├── processed_data                      # Aggregated processed data in keyword .txt files and .csv
    ├── processing                          # Scraping and processing code
    ├── automatic_data_processing.py        # Script to run scrapping and processing for all conferences
    ├── manual_data_processing.py           # Script to aggregate the manual annotations for 2024
    ├── play_with_automated_data.ipynb      # Notebook for plots of the automated processing
    ├── play_with_manual_data.ipynb         # Notebook for plots of the manual processing
    ├── play_with_keywords.ipynb            # Notebook for wordclouds of the keywords
    ├── LICENSE
    └── README.md

### Usage:

To replicate the scraping and processing, first install the dependencies:

```python
pip install uv
make install
```

To then run all processing, use:
```python
make process
```

Be warned: this will download all papers from 2018-2024, so it takes quite a while! 
You can do this if you want to replicate the full process, but you don't need to if you just want to look into the data.
We provide the fully processed data including the plotting scripts, so you can simply use the notebooks to get more detailed insights.
Note that we do not include the keyword visualization in our arguments, but they might be interesting to you nonetheless.