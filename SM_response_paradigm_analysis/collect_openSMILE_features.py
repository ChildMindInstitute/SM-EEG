#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_openSMILE_features.py

Functions to collect openSMILE features from a collection of sound files for
analysis.

Author:
    – Jon Clucas, 2017 (jon.clucas@childmind.org)

© 2017, Child Mind Institute, Apache v2.0 License
"""
import argparse, os, pandas as pd, subprocess
from configuration_defaults import openSMILE_directory


def build_table(oS, audio, config):
    """
    Build table with openSMILE features

    Parameters
    ----------
    oS : string
        path to openSMILE binary

    audio : string
        path to audio files

    config : string
        path to openSMILE config file

    Returns
    ------
    table : pandas dataframe
        one row per trial, columns for URSI, stranger, trial, and each feature
    """
    temp = os.path.join(audio, "temp.csv")
    table = pd.DataFrame()
    for wav_file in os.listdir(audio):
        if ("vocal" in wav_file and "button" not in wav_file and
            wav_file.endswith(".wav")):
                ursi = wav_file[:9]
                condition, trial = wav_file[-13:-4].lstrip("_").split("_", 1)
                oScommand = oS + " -C " + config + " -I " + os.path.join(audio,
                        wav_file) + " -O " + temp
                print(oScommand)
                subprocess.call(oScommand, shell=True)
                features = read_temp(temp)
                os.remove(temp)
                try:
                    trial_no = str(int(trial[-3:]))
                except:
                    trial_no = str(int(wav_file.split("exp")[1].rstrip(".wav"))
                               )
                table = table.append({"URSI": ursi.upper(), "stranger":
                        condition, "observation": trial_no, **features},
                        ignore_index=True)
    return(table.sort_values(["URSI", "stranger", "observation"]))


def main():
    """
    Set up and run openSMILE

    Parameters
    ----------
    (from command line)
    audio : string
        path to directory with audio files to process

    config : string
        filename of openSMILE config file

    outdir : string (optional)
        path to directory in which to save openSMILE outputs, default=
        [openSMILE_config] in parent directory of [audio]

    openSMILE_directory : string (optional)
        path to openSMILE installation, default="../../../openSMILE-2.3.0"

    Output
    ------
    features.csv : comma separated values file
        one row per trial
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", metavar="audio_directory", type=str, help=
                        "path to directory with audio files to process")
    parser.add_argument("config", metavar="openSMILE_config", type=str, help=
                        "filename of openSMILE config file")
    parser.add_argument("-o", "--output", metavar="output_directory", dest=
                        "outdir", type=str, help=("path to directory in which "
                        "to save openSMILE outputs, default=[openSMILE_config]"
                        "in parent directory of [audio]"))
    parser.add_argument("-S", "--openSMILE", default=openSMILE_directory,
                        metavar="openSMILE", dest="openSMILE_directory", type=
                        str, help=("path to openSMILE installation, default=\""
                        "../../../openSMILE-2.3.0\""))
    arg = parser.parse_args()
    outdir = arg.outdir if arg.outdir else os.path.join(os.path.dirname(
             arg.audio.rstrip("/")), arg.config)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    table = build_table(os.path.join(arg.openSMILE_directory, "inst", "bin",
            "SMILExtract"), arg.audio, os.path.join(arg.openSMILE_directory,
            "config", arg.config))
    print(table)
    table.to_csv(os.path.join(outdir, "features.csv"))


def read_temp(temp):
    """
    Read openSMILE features for a single file

    Parameters
    ----------
    temp : string
        path to openSMILE feature ARFF csv

    Returns
    ------
    features : dictionary
        dictionary of features from openSMILE output
    """
    type_dict = {"string": str, "unknown": str, "numeric": float}
    with open(temp, 'r') as topen:
        feature_lines = topen.readlines()
    feature_labels = []
    feature_types = []
    for i, row in enumerate(feature_lines):
        if row.startswith("@attribute"):
            flabel, ftype = row[11:-1].split(' ')
            feature_labels.append(flabel)
            feature_types.append(type_dict[ftype])
        elif row.startswith("@data"):
            feature_values = feature_lines[i+1].split(",")
            if len(feature_values) < len(feature_labels):
                feature_values = feature_lines[i+2].split(",")
    for i, item in enumerate(feature_values):
        try:
            feature_values[i] = (feature_types[i](item))
        except:
            feature_values[i] = item
    return(dict(zip(feature_labels, feature_values)))

# ============================================================================
if __name__ == '__main__':
    main()
