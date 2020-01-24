__author__ = 'lucabasa'
__version__ = '2.0.0'
__status__ = 'development'

import pandas as pd
import numpy as np
import string
import random

from sklearn.datasets import make_regression


def make_uncorrelated(data, n_entries):
    df = data.copy()
    np.random.seed(23)
    random.seed(23)
    # normal distributions
    ran_loc = np.random.normal(loc=1, scale=0.8, size=20)
    ran_scl = np.random.uniform(low=0.01, high=3, size=20)
    for i in range(1,21):
        df[f'unc_normal_{i}'] = np.random.normal(loc=ran_loc[i-1], scale=ran_scl[i-1], size=n_entries)
        
    # Skewed distribution
    df['unc_skewed_pos'] = np.expm1(np.random.normal(loc=1.3, scale=0.5, size=n_entries))
    df['unc_skewed_neg'] = 40 - np.expm1(np.random.normal(loc=2.3, scale=0.3, size=n_entries))
    
    # categorical variables
    df['unc_binary_1'] = np.random.choice([0, 1], size=(n_entries,), p=[1/3, 2/3])
    df['unc_binary_2'] = np.random.choice([0, 1], size=(n_entries,), p=[1/2, 1/2])
    df['unc_binary_3'] = np.random.choice([0, 1], size=(n_entries,), p=[1/5, 4/5])
    df['unc_binary_4'] = np.random.choice([0, 1], size=(n_entries,), p=[1/4, 3/4])
    df['unc_binary_5'] = np.random.choice([0, 1], size=(n_entries,), p=[1/10, 9/10])
    
    df['unc_categories_5'] = np.random.choice(['a', 'b', 'c', 'd', 'e'], size=(n_entries,), p=[2/5, 1/5, 1/10, 3/15, 1/10])
    df['unc_categories_3'] = np.random.choice(['a', 'b', 'c'], size=(n_entries,))
    
    for i in range(1,12):
        df[f'unc_ordinal_{i}'] = np.random.choice(np.arange(1,i*10), size=(n_entries,))
    
    return df


def make_correlated(data, n_entries):
    df = data.copy()
    np.random.seed(23)
    random.seed(23)
    # correlation with 1 category
    df['corr_cat_1'] = np.random.choice([0, 1], size=(n_entries,), p=[1/3, 2/3])
    df['corr_cat_2'] = np.random.choice([0, 1], size=(n_entries,))
    df.loc[df.corr_cat_1 == 0, 'corr_normal_by_cat'] = np.random.normal(loc=-0.5, scale=0.4, size=n_entries - df.corr_cat_1.sum())
    df.loc[df.corr_cat_1 == 1, 'corr_normal_by_cat'] = np.random.normal(loc=1, scale=0.2, size=df.corr_cat_1.sum())
    
    # correlation with 2 categories
    tmp = df.groupby(['corr_cat_1', 'corr_cat_2']).size()
    df.loc[(df.corr_cat_1 == 0) & (df.corr_cat_2 == 0) , 'corr_normal_by_2cats'] = np.random.normal(loc=2, scale=0.4, size=tmp[0][0])
    df.loc[(df.corr_cat_1 == 0) & (df.corr_cat_2 == 1) , 'corr_normal_by_2cats'] = np.random.normal(loc=4, scale=0.65, size=tmp[0][1])
    df.loc[(df.corr_cat_1 == 1) & (df.corr_cat_2 == 0) , 'corr_normal_by_2cats'] = np.random.normal(loc=1, scale=1, size=tmp[1][0])
    df.loc[(df.corr_cat_1 == 1) & (df.corr_cat_2 == 1) , 'corr_normal_by_2cats'] = np.random.normal(loc=-1, scale=0.5, size=tmp[1][1])
    
    # multinormals with different correlations
    df['corr_multinormal_high_a'] = 0
    df['corr_multinormal_high_b'] = 0
    df[['corr_multinormal_high_a', 'corr_multinormal_high_b']] = np.random.multivariate_normal(mean=[6, -2], cov=[[2, -0.8], [-0.8, 0.5]], size=n_entries)
    
    df['corr_multinormal_mid_a'] = 0
    df['corr_multinormal_mid_b'] = 0
    df[['corr_multinormal_mid_a', 'corr_multinormal_mid_b']] = np.random.multivariate_normal(mean=[20, 11], cov=[[7, 2.9], [2.9, 5]], size=n_entries)
    
    df['corr_multinormal_low_a'] = 0
    df['corr_multinormal_low_b'] = 0
    df[['corr_multinormal_low_a', 'corr_multinormal_low_b']] = np.random.multivariate_normal(mean=[-6, 4], cov=[[1, 0.3], [0.3, 2]], size=n_entries)
    
    df[['corr_multinormal_mid_a', 'corr_multinormal_mid_b', 
        'corr_multinormal_low_a', 'corr_multinormal_low_b']] += np.random.multivariate_normal(mean=[0, 0, 0, 0], 
                                                                                              cov=[[2, -0.8, -1.2, 0.02], 
                                                                                                   [-0.8, 0.5, 0, 0], 
                                                                                                   [-1.2, 0, 7, 0.9], 
                                                                                                   [0.02, 0, 0.9, 5]], size=n_entries)
    
    return df


def make_confounding(data):
    df = data.copy()
    np.random.seed(23)
    random.seed(23)
    n_entries = data.shape[0]
    
    # normal distributions uncorrelated from the rest
    ran_loc = np.random.normal(loc=0, scale=1, size=40)
    ran_scl = np.random.uniform(low=0.01, high=5, size=40)
    for i in range(1,41):
        df[f'conf_normal_{i}'] = np.random.normal(loc=ran_loc[i-1], scale=ran_scl[i-1], size=n_entries)
        
    # interactions of other variables with small noise
    for i in range(1,6):
        df[f'conf_inter_{i}'] = df[f'unc_normal_{i}'] * df[f'unc_normal_{i+10}'] + np.random.normal(loc=0, scale=0.2, size=n_entries)
        
    df['conf_inter_6'] = df['unc_skewed_pos'] * df['corr_multinormal_high_a'] + np.random.normal(loc=0, scale=0.1, size=n_entries)
    df['conf_inter_7'] = df['unc_binary_4'] * df['unc_binary_2'] + np.random.normal(loc=1, scale=0.1, size=n_entries)
    df['conf_inter_8'] = df['unc_binary_1'] * df['unc_binary_2'] + np.random.normal(loc=0.1, scale=0.2, size=n_entries)
    df['conf_inter_9'] = df['corr_multinormal_mid_b'] * df['corr_multinormal_low_b'] + np.random.normal(loc=-0.1, scale=0.1, size=n_entries)
    df['conf_inter_10'] = df['corr_multinormal_low_a'] * df['unc_normal_19'] + np.random.normal(loc=-1, scale=0.2, size=n_entries)
    
    return df


def linear_targets(data, tmp):
    df = data.copy()
    np.random.seed(2332)
    random.seed(234)
    entries = df.shape[0]
    
    coef_dict = {}
    
    # all
    coefs = np.random.normal(loc=0, scale=0.5, size=tmp.shape[1])
    df['tar_lin_full'] = tmp.multiply(coefs, axis=1).sum(axis=1) + np.random.normal(20, 10, size=entries)
    coef_dict['tar_lin_full'] = pd.DataFrame({'feat': tmp.columns, 'coef': coefs})
    
    # with half of them
    to_use = random.sample([col for col in tmp.columns if 'conf_' not in col], 50)
    coefs = np.random.normal(loc=0, scale=2, size=len(to_use))
    df['tar_lin_half'] = tmp[to_use].multiply(coefs, axis=1).sum(axis=1) + np.random.normal(-20, 10, size=entries)
    coef_dict['tar_lin_half'] = pd.DataFrame({'feat': to_use, 'coef': coefs})
    
    # with 10 of them
    to_use = random.sample([col for col in tmp.columns if 'conf_' not in col and 'categories' not in col], 10)
    coefs = np.random.normal(loc=0, scale=0.5, size=len(to_use))
    df['tar_lin_10'] = tmp[to_use].multiply(coefs, axis=1).sum(axis=1) + np.random.normal(0, 1, size=entries)
    coef_dict['tar_lin_10'] = pd.DataFrame({'feat': to_use, 'coef': coefs})
    
    return df, coef_dict


def nonlinear_targets(data, tmp):
    df = data.copy()
    np.random.seed(23)
    entries = df.shape[0]
    
    coef_dict = {}
    
    df['tar_nonlin_10'] = -200 + (0.3 * df['unc_normal_1']**2 
                           -0.7 * df['unc_normal_7']**2 
                           -1 * df['unc_skewed_pos'] * df['unc_skewed_neg'] 
                           +0.8 * df['corr_multinormal_mid_a']**2 
                           +0.5 * df['unc_ordinal_5'] * df['unc_normal_12'] 
                           -0.4 * df['unc_normal_14'] * df['corr_normal_by_cat'] 
                           + df['unc_normal_1'] * df['unc_normal_7']
                          )
    coef_dict['tar_nonlin_10'] = pd.DataFrame({'feat': ['unc_normal_1_sq', 
                                                        'unc_normal_7_sq', 
                                                        'unc_skewed_int', 
                                                        'corr_multinormal_mid_a_sq', 
                                                        'unc_ordinal_5_12', 
                                                        'unc_normal_14_by_cat', 
                                                        'unc_normal_1_7'],
                                               'coef': [0.3, -0.7, -1, 0.8, 0.5, -0.4, 1]})
    
    return df, coef_dict


def make_targets(data):
    tmp = data.copy()
    tmp = pd.get_dummies(tmp, drop_first=True)
    
    df, coef_lin = linear_targets(data, tmp)
    df, coef_nonlin = nonlinear_targets(df, tmp)

    return df, coef_lin, coef_nonlin


def dirtify(data):
    df = data.copy()
    n_entries = df.shape[0]
    np.random.seed(11)
    
    # adding random noise to some columns
    df['unc_normal_1'] += np.random.normal(loc=0, scale=15, size=n_entries) / 10
    df['corr_normal_by_cat'] += np.random.normal(loc=-0.5, scale=5, size=n_entries) / 10
    df['unc_skewed_neg'] += np.random.normal(loc=0, scale=50, size=n_entries) / 10
    
    
    # make outliers
    random_entries = np.random.random(df.shape[0])<0.00001
    df['unc_normal_2'] = df['unc_normal_2'].mask(random_entries).fillna(df['unc_normal_2'] + 20)
    random_entries = np.random.random(df.shape[0])<0.00001
    df['corr_multinormal_mid_a'] = df['corr_multinormal_mid_a'].mask(random_entries).fillna(df['corr_multinormal_mid_a'] - 54)
    
    return df


def make_data_regression(n_features, n_informative, noise, effective_rank=None, tail_stregth=0.5):
    data_sim, target, coefficients = make_regression(n_samples=100000, 
                                                 n_features=n_features, n_informative=n_informative, 
                                                 effective_rank=effective_rank, tail_strength=tail_stregth,
                                                 coef=True, random_state=21, noise=noise)
    
    random.seed(23)
    i = 0
    random_names = []
    # generate n_features random strings of 5 characters
    while i < n_features:
        random_names.append(''.join([random.choice(string.ascii_lowercase) for _ in range(5)]))
        i += 1
    
    data_sim = pd.DataFrame(data_sim, columns=random_names)
    data_sim['target'] = target

    coeff = pd.DataFrame({'variable': random_names, 'coefficient': coefficients})
    
    return data_sim, coeff
