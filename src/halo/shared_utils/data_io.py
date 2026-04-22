import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def classify_interaction(bliss, additivity_cutoff=0.1):
    """
    Adding classification labels based on the additivity cutoff and Bliss scores.
    Bliss score < -0.12 -> synergy
    -0.12 < Bliss score < 0.12 -> additivity
    Bliss score > 0.12 -> antagonism

    Returns:
        labels for `Interaction Type` column of df
    """
    if bliss <= -(additivity_cutoff):
        return 'synergy'
    elif -(additivity_cutoff) < bliss < additivity_cutoff:
        return 'neutral'
    if bliss >= additivity_cutoff:
        return 'antagonism'


def features_and_target(df: pd.DataFrame, 
                        task: str,                   # 'regr' | 'bin_clas' | 'multi_clas'
                        strain_as_feature: bool,     # True if Strain is considered as one categorical feature
                        top_n_strains: int = None    # optional: filter to top N most frequent strains, if None it includes all strains
                        ):
    """
    Splits df into X (features) and y (target) depending on task type.
    Optionally adds Strain as a categorical feature also limits data to top-N strains.

    Returns:
        X (pd.Dataframe): feature set
        y (Numpy array): of Bliss scores (regression task) or encoded labels (classification task) 
    """
    # ---- filtering df to binary task if needed ----
    if task == 'bin_clas':
        df = df[df['Interaction Type'].isin(['synergy', 'antagonism'])].copy() # binary df
    elif task in ('multi_clas', 'regr'):
         df = df.copy() 
    else: 
        raise ValueError("Task must be one of: 'bin_clas', 'multi_clas', 'regr'.")
    

    # ----- features -----
    if top_n_strains is not None:
        top_strains = df['Strain'].value_counts().head(top_n_strains).index
        df = df[df['Strain'].isin(top_strains)].copy()

    if task == 'bin_clas':
        class_counts = df['Interaction Type'].value_counts()
    if class_counts.size < 2:
        raise ValueError("After filtering, only one class remained.")

    
    metadata_cols = ['Drug A', 'Drug B', 'Drug A Inchikey', 'Drug B Inchikey', 'Strain', 'Specie', 
                     'Drug Pair', 'Source', 'Bliss Score', 'Interaction Type']
    feat_cols = [c for c in df.columns if c not in metadata_cols]


    if strain_as_feature:
        # create an integer code for Strain so all tree models can eat it
        # pd.factorize returns (codes, uniques) where codes are int64
        strain_codes, strain_uniques = pd.factorize(df['Strain'], sort=True)
        df['Strain_code'] = strain_codes.astype('int64')
        df['Strain'] = df['Strain'].astype('category')
        # we do NOT keep the raw string 'Strain' in features, we only add the numeric code
        feat_cols = feat_cols + ['Strain_code']


    X = df[feat_cols].copy()

    # ----- target -----
    class_names = None

    if task in ('multi_clas', 'bin_clas'):
        y = df['Interaction Type']
        le = LabelEncoder()
        y = le.fit_transform(y)
        class_names = le.classes_
    elif task == 'regr':
        y = df['Bliss Score'].values
    else:
        raise ValueError(f"Unknown task: {task}")

    return X, y, class_names



def basic_split(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Simple X, y splitting

    Returns:
        X_train
        X_test
        y_train 
        y_test
    """

    if stratify:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)




    







