import csv
import datetime
import dateutil.tz
import json
import pathlib

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from drl_algos import utils

# Sub-directory within deepRLalgos repo for storing logs
# e.g., deepRLalgos/experiments/
LOGGING_DIR = "experiments"

# Find (and create) full path to LOGGING_DIR
base_dir = pathlib.Path(__file__).parent.parent.parent
exp_dir = base_dir.joinpath(LOGGING_DIR)
if not exp_dir.exists():
    exp_dir.mkdir()

"""
Changelog
    - Now works with None values
"""


class Logger2(object):
    """Logger for recording and reporting training runs.

    args:
        log_dir (String): directory within LOGGING_DIR (see above) for storing
            logs. The directory name will be timestamped making it semi-unique

    Note:
        - tensorflow may complain about not being able to find the right version
        of cuda but this warning can be ignored as tensorboard does not need
        cuda.
    """

    def __init__(self, log_dir):
        self.log_dir = exp_dir.joinpath(log_dir + "_" + self._get_exp_name())

        self.log_dir.mkdir(parents=True)
        self.checkpoint_dir = self.log_dir.joinpath("checkpoints")
        self.checkpoint_dir.mkdir()
        self.writer = SummaryWriter(self.log_dir)
        self.file_path = self.log_dir.joinpath("stats.csv")

    def log(self, timestep, stats_dict):
        """Logs dictionary of stats under given timestep in a csv file.

        Important:
            - Always call with the complete set of keys passing None for any
            keys without a value. Calling with variable number of keys will
            result in an incorrect csv file.

        Args:
            timestep (int): timestep in training run
            stats_dict (dict): dictionary to log to
        """
        # Construct csv key-values
        keys = ["timestep"]
        values = [timestep]
        for key in sorted(stats_dict.keys()):
            keys.append(key)
            value = stats_dict[key]
            values.append(value)
            # Write to tensorboard
            if value is not None:
                self.writer.add_scalar(key, value, timestep)

        # Write to csv file
        self._write_csv(keys, values, self.file_path)

        # Print to the console
        self._print_tabular_vert(keys, values)

    def save_config(self, config):
        """Saves the given config as a json file.

        Args:
            congif (dict): dictionary detailing the training run.
        """
        file_path = self.log_dir.joinpath("config.json")
        with open(file_path, "w") as config_file:
            json.dump(config, config_file, indent=4)

    def save_params(self, timestep, network_dict):
        """Saves the dictionary of networks.

        Args:
            timestep (int): timestep in training run
            network_dict (dict): dictionary of networks to save.
        """
        file_path = self.checkpoint_dir.joinpath("%s.pkl" % timestep)
        torch.save(network_dict, file_path)

    def _print_tabular_horiz(self, keys, values):
        """Nicely prints a horizontal table."""
        header = "|"
        row = "|"
        divider = "+"
        divider2 = "+"
        for key, value in zip(keys, values):
            # Calculate column width and value type, int format works for None
            if isinstance(value, int) or value is None:
                isInt = True
                valueLen = len(str(value))
            else:
                isInt = False
                valueLen = len("%.3f" % value)
            columnWidth = max(valueLen, len(key))

            # Format header, data and divider rows
            header += " %" + str(columnWidth) + "s |"
            if isInt:
                row += " %" + str(columnWidth) + "s |"
            else:
                row += " %" + str(columnWidth) + ".3f |"
            divider += "-" * (columnWidth + 2) + "+"

        # Print the table
        print()
        print(divider)
        print(header % (*keys,))
        print(divider)
        print(row % (*values,))
        print(divider)
        print()

    def _print_tabular_vert(self, keys, values):
        """Nicely prints a vertical table."""
        # Find longest key
        longest_key = 0
        for key in keys:
            key_len = len(key)
            if key_len > longest_key:
                longest_key = key_len

        # Format header and the key portion of each row
        header = ("+"
                  + "-" * (longest_key + 2) + "+"
                  + "-" * 12 + "+")
        row = "| %-" + str(longest_key) + "s | "

        # Print the table
        print()
        print(header)
        for i in range(len(keys)):
            key = keys[i]
            value = values[i]
            if value is None:
                full_row = row + "%-10s |"
            elif value - int(value) == 0 and len(str(value)) < 10:
                if value < 0:
                    full_row = row + "%-10d |"
                else:
                    full_row = row + "%-9d  |"
            else:
                str_len = len(str("%.3f" % value))
                if str_len < 10:
                    white_space = 10 - str_len
                    full_row = row + "%.3f " + (" " * white_space) + "|"
                else:
                    if value < 0:
                        full_row = row + "%.3e |"
                    else:
                        full_row = row + "%.3e  |"
            print(full_row % (key, value))
            if i == 0:
                print(header) # Highlights the timestep when printing
        print(header)

    def _write_csv(self, keys, values, file):
        """Saves key-values to the specified csv file.

        Not designed to handle NaNs.
        """
        if pathlib.Path(file).is_file():
            # Append new row to the existing csv file
            with open(file, "a") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(values)
        else:
            # Create new csv file then add a header and the row
            with open(file, "w") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(keys)
                csv_writer.writerow(values)

    def _get_exp_name(self):
        """Gets timestamp in format Year-Month-Day-Hour-Minute-Second."""
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        return now.strftime('%Y_%m_%d_%H_%M_%S')
