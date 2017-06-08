#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 16:24:00 2017

@author: jon.clucas
"""

import os, pandas as pd

os.chdir("/Users/jon.clucas/audio_files")

def main():
    all_URSIs = set()
    adults_speak = set()
    extra_kid = pd.DataFrame(list(set([("M00453327", True, True), ("M00453327",
                False, True)])), columns=["URSI", "stranger", "extra_child"])
    all_URSIs = pd.DataFrame(list(update_set(all_URSIs, "nobeeps")), columns=[
                "URSI", "stranger"])
    adults_speak = pd.DataFrame(list(update_set(adults_speak, "adults")),
                   columns=["URSI", "stranger"])
    adults_speak["adults_speak"] = True
    table = pd.merge(pd.merge(all_URSIs, adults_speak, how="outer", on=["URSI",
            "stranger"]).fillna(False), extra_kid, how="outer", on=["URSI",
            "stranger"]).fillna(False)
    table.to_csv("participants_conditions.csv", index=False)
    
def update_set(s, d):
    """
    function to update a set with URSIs and stranger presence from
    vocal-condition wav files in a given directory
    
    Parameters
    ----------
    s : set
        set to update
        
    d : string
        path to directory to update from
        
    Returns
    -------
    s : set
        updated set
    """
    for wav in os.listdir(d):
        if "vocal" in wav:
            if "_w_" in wav:
                s.add((wav[:9].upper(), True))
            elif "_no_" in wav:
                s.add((wav[:9].upper(), False))
            else:
                print(wav)
    return s

# ============================================================================
if __name__ == '__main__':
    main()