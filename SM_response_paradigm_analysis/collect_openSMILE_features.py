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
    Build and save

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
                condition, trial = wav_file[-13:-4].lstrip("_").split("_")
                oScommand = oS + " -C " + config + " -I " + os.path.join(audio,
                        wav_file) + " -O " + temp
                print(oScommand)
                subprocess.call(oScommand, shell=True)
                features_all = pd.read_csv(temp, sep=None)
                os.remove(temp)
                features = dict(zip(features_all.iloc[:-2][
                           features_all.columns[-1]], features_all.iloc[-1
                           ].values[1].split(",")))
                table = table.append({"URSI": ursi, "stranger": condition,
                        "trial": str(int(trial[-3:])), **features},
                        ignore_index=True)
    return(table)


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


# ============================================================================
if __name__ == '__main__':
    main()
