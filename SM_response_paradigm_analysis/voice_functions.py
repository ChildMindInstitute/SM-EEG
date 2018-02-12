#!/usr/bin/env python
import numpy as np
import pandas as pd

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
    ycols = ['Dx?']
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