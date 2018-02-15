#!/usr/bin/env python
import json
import numpy as np
import matplotlib as plt
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sys
sm_rpa_v = os.path.abspath(os.getcwd())
while not os.path.exists(
    os.path.join(
        sm_rpa_v,
        "SM_response_paradigm_analysis"
    )
):
    sm_rpa_v = os.path.abspath(
        os.path.join(
            sm_rpa_v,
            os.pardir
        )
    )
if sm_rpa_v not in sys.path: 
    sys.path.append(sm_rpa_v)
from SM_openSMILE.openSMILE_analysis import openSMILE_csv
with open(
    os.path.join(
        sm_rpa_v,
        'config/config.json'
    )
) as cfgf:
    osf = json.load(cfgf)['OSF_urls']


def combine_data(openSMILE, conditions, dx):
    """
    Function to combine imported DataFrames.
    
    Parameters
    ----------
    openSMILE: pandas DataFrame
        features from openSMILE
    
    conditions: pandas DataFrame
        notes about each recording
        
    dx: pandas DataFrame
        diagnostic information
        
    Returns
    -------
    d2: pandas DataFrame
        merged DataFrame
    """
    openSMILE['stranger'] = openSMILE['stranger'] == 'w'
    d = pd.merge(dx, openSMILE, on='URSI', how='right')
    d2 = pd.merge(d, conditions, on=['URSI', 'stranger'], how='left')
    return(d2)


def feature_names(df, config_file):
    """
    Function to gather and apply column names to an openSMILE dataframe
    
    Parameters
    ----------
    df: DataFrame
        with columns ["X", "Y"]
    
    config_file: str
        basename for openSMILE config file, eg, ComParE_2016
        
    Returns
    -------
    df: DataFrame
        with appropriate column headers
        ("Y" is now "Selective Mutism diagnosis")
    """
    smdx = df["Y"]
    rows = [
        row for row in df["X"]
    ]
    df = pd.DataFrame(
        {
            i: rows[
                i
            ] for i in range(
                len(
                    rows
                )
            )
        }
    ).T
    with open(
        os.path.join(
            sm_rpa_v,
            "SM_openSMILE/openSMILE_preprocessing/openSMILE_outputs",
            config_file,
            "full_original.csv"
        )
    ) as f:
        arf_features=f.readlines()
    df.columns = [
        f[
            len(
                "@attribute"
            )+1:
        ].split(
            " "
        )[
            0
        ] for f in arf_features if f.startswith(
            "@attribute"
        )
    ]
    df["Selective Mutism diagnosis"] = [
        'SM' == "".join(
            y
        ) for y in smdx
    ]
    return(df)


def int_categorize(df):
    """
    Function to take a dataframe with some categorical data and
    to reencode those categories as integers.
    
    Parameter
    ---------
    df : DataFrame
        DataFrame to reencode
        
    Returns
    -------
    df : DataFrame
        Reencoded DataFrame
    """
    if "Dx?" in df.columns:
        df["Dx?"] = df["Dx?"].fillna(False).astype(bool)
    up = []
    for c in list(df.columns):
        if(str(df[c].dtype) == "object"):
            up.append(c)
    dicts = [dict() for u in up]
    df = update_encoding(df, dicts, up, 'category')
    for u in up:
        df = update_encoding(
                df,
                {m: i for i, m in enumerate(list(df[u].cat.categories))},
                u,
                int)
    return(df)


def load_from_osf(config_file, experimental_condition, noise_replacement):
    """
    
    Parameters
    ----------
    config_file: str
        basename of openSMILE configuration file, eg `emobase`
        
    experimental_condition: str
        experimental condition, eg, `voice, no stranger`
        
    noise_replacement: str
        noise replacement condition, eg, `adults replaced: clone`
    """
    return(
        feature_names(
            pd.DataFrame(
                openSMILE_csv.get_features(
                    osf[
                        config_file
                    ][
                        experimental_condition
                    ][
                        noise_replacement
                    ],
                    config_file
                ),
                columns=[
                    "X",
                    "Y"
                ]
            ),
            config_file
        )
    )


def make_forest(configed_df):
    """
    Function to get training and target data, filling in unaltered rows when no
    altered row exists

    Parameters
    ----------     
    configed_df : pandas DataFrame
        DataFrame with openSMILE output and demographic features

    Returns
    -------
    x_trees : numpy array
        array of [n_participants × n_features] size
        filled with training data (features)

    y_trees : numpy array
        array of [n_participants × n_dx_features] size
        filled with target data (diagnoses)
    """
    # ycols = [col for col in configed_df.columns if ('smq' in col)] + ['Dx?','URSI']
    ycols = ['Selective Mutism diagnosis']
    xcols = configed_df.columns.difference(ycols)
    for col in xcols:
        try:
            float(configed_df[col][0])
        except:
            enc = LabelEncoder()
            enc.fit(np.array(configed_df[col]))
            configed_df[col] = enc.transform(np.array(configed_df[col]))
    xtrees = np.array(configed_df[xcols]).reshape(len(configed_df), len(xcols))
    ytrees = np.array(configed_df[ycols]).reshape(len(configed_df), len(ycols)).ravel()
    return (xtrees, ytrees)


def reencode(df, mapping, field, dtype=None):
    """
    Private function to assist `update_encoding` to update encodings
    in a given DataFrame or list of DataFrames.
    
    Parameters
    ----------
    df : pandas DataFrame or list thereof
        DataFrame to correct data in
    
    mapping : dictionary or list thereof
        {incorrect:correct} to be applied to (all) given DataFrame(s)
        
    field : string
        the column name or list thereof to update in the given DataFrame(s);
        if a list, must be the same length as mapping and in the same order
        
    Returns
    -------
    df(2) : pandas DataFrame or list thereof
        same shape as input, but with corrected URSIs
    """
    if field in df.columns:
        mapping = {**mapping, **{ursi:ursi for ursi in df[field] if ursi not in mapping}}
        if dtype:
            df[field] = df[field].map(mapping).astype(dtype)
        else:
            df[field] = df[field].map(mapping)
    return(df)


def SM_forest(all_dict, config_file, condition, noise_replacement, ntrees=2000):
    """
    Function to run random forests on one
    config file × exprimental condition × noise replacement method
    
    Parameters
    ----------
    all_dict: dictionary
        [config][experimental condition][noise replacement method]["DataFrame"] keys
        
    config_file: str
        openSMILE config file basename
        
    condition: str
        experimental condition
        
    noise_replacement: str
        noise replacement method
        
    ntrees: int, optional
        number of 🌲s, default=2,000
        
    Returns
    -------
    features: DataFrame
        features ranked by importance
        
    model: RandomForestClassifier
        fit model
    """
    X = int_categorize(
        all_dict[
            config_file
        ][
            condition
        ][
            noise_replacement
        ][
            "DataFrame"
        ].loc[
            :,
            all_dict[
                config_file
            ][
                condition
            ][
                noise_replacement
            ][
                "DataFrame"
            ].columns.difference([
                "Selective Mutism diagnosis"
            ])
        ]
    )
    Y = all_dict[
        config_file
    ][
        condition
    ][
        noise_replacement
    ][
        "DataFrame"
    ][
        "Selective Mutism diagnosis"
    ]
    clf = RandomForestRegressor(
        n_estimators=ntrees,
        oob_score = True
    )
    clf = clf.fit(
        X,
        Y
    )
    features = pd.DataFrame.from_dict(
        dict(
            zip(
                X,
                clf.feature_importances_
            )
        ),
        orient='index'
    ).rename(
        columns={
            0:"importance"
        }
    ).sort_values(
        "importance",
        ascending=False
    )
    print(
        "Most predictive feature for {0} config file "
        "× {1} × {2} was {3} with an importance score "
        "of {4}".format(
            config_file,
            condition,
            noise_replacement,
            features.ix[0].name,
            features.ix[0].importance
        )
    )
    return(
        features,
        clf
    )


def update_encoding(df, mapping, field, dtype=None):
    """
    Function to update encodings in a given DataFrame or list of DataFrames.
    
    Parameters
    ----------
    df : pandas DataFrame or list thereof
        DataFrame to correct data in
    
    mapping : dictionary or list thereof
        {incorrect:correct} to be applied to (all) given DataFrame(s)
        
    field : string
        the column name or list thereof to update in the given DataFrame(s);
        if a list, must be the same length as `mapping` and in the same order
        
    dtype : type (optional)
        datatype or list thereof to recast the given column; like `field`,
        if a list, must be the same length as `mapping` and in the same order
        
    Returns
    -------
    df(2) : pandas DataFrame or list thereof
        same shape as input, but with corrected URSIs
    """
    if(type(df) == list):
        df2 = list()
        for d in df:
            df2.append(update_encoding(d, mapping, field))
        return(df2)
    else:
        df = df.drop(
            "Unnamed: 0",
            axis=1
        ) if "Unnamed: 0" in df.columns else df
        if(type(mapping) == list):
            for i, m in enumerate(mapping):
                df = reencode(df, m, field[i], dtype)
        else:
            df = reencode(df, mapping, field, dtype)
        return(df)