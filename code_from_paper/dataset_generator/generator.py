"""
THIS IS THE MAIN GENERATOR
"""
import sys
sys.path.insert(0,'/Users/rahat/Documents/GitHub/ics691d_bci/code_from_paper') # Change this to set the base directory of project
from data_processing.general_processor import Utils
import numpy as np
import os

channels = [["FC1", "FC2"],
            ["FC3", "FC4"],
            ["FC5", "FC6"],
            ["C5", "C6"],
            ["C3", "C4"],
            ["C1", "C2"],
            ["CP1", "CP2"],
            ["CP3", "CP4"],
            ["CP5", "CP6"]]

exclude = [38, 88, 89, 92, 100, 104]
subjects = [n for n in np.arange(1, 110) if n not in exclude]
runs = [5, 6, 9, 10, 13, 14]
data_path = "/Users/rahat/Downloads/eegdatabci/" # Path to original files
for couple in channels:
    base_path = "/Users/rahat/Downloads/eegdatabci/processed_files_run2/" # Path to processed files
    save_path = os.path.join(base_path, couple[0] + couple[1])
    os.makedirs(save_path, exist_ok=True)
    for sub in subjects:
        print(f"Subject: {sub}")
        x, y = Utils.epoch(Utils.select_channels
            (Utils.eeg_settings(Utils.del_annotations(Utils.concatenate_runs(
            Utils.load_data(subjects=[sub], runs=runs, data_path=data_path)))), couple),
            exclude_base=False)

        np.save(os.path.join(save_path, "x_sub_" + str(sub)), x, allow_pickle=True)
        np.save(os.path.join(save_path, "y_sub_" + str(sub)), y, allow_pickle=True)
