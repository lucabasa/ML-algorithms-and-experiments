__author__ = 'lucabasa'
__version__ = '1.2.0'
__status__ = 'development'

import pandas as pd
import numpy as np
import string
import random


def make_uncorrelated(data, n_entries):
    df = data.copy()
    np.random.seed(23)
    random.seed(23)
    # normal distributions
    df['unc_normal_1'] = np.random.normal(loc=0, scale=0.5, size=n_entries)
    df['unc_normal_2'] = np.random.normal(loc=10, scale=1, size=n_entries)
    # Skewed distribution
    df['unc_skewed_pos'] = np.expm1(np.random.normal(loc=1.3, scale=0.5, size=n_entries))
    df['unc_skewed_neg'] = 40 - np.expm1(np.random.normal(loc=2.3, scale=0.3, size=n_entries))
    
    # categorical variables
    df['unc_binary'] = np.random.choice([0, 1], size=(n_entries,), p=[1/3, 2/3])
    df['unc_categories_5'] = np.random.choice(['a', 'b', 'c', 'd', 'e'], size=(n_entries,), p=[2/5, 1/5, 1/10, 3/15, 1/10])
    i = 0
    random_cats = []
    # generate 100 random strings of 3 categories
    while i < 100:
        random_cats.append(''.join([random.choice(string.ascii_lowercase) for _ in range(3)]))
        i += 1
    df['unc_categories_100'] = np.random.choice(random_cats, size=(n_entries,))
    
    df['unc_ordinal'] = np.random.choice(np.arange(1,100), size=(n_entries,))
    
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


def linear_targets(data, tmp):
    df = data.copy()
    np.random.seed(23)
    entries = df.shape[0]
    
    coef_dict = {}
    
    # all
    coefs = [3, 0.4, -0.01, 0.03, -2.4, 0.11, 0.1, 6, 2.36, 7.3, 0.75, 0.69, -9.47, 0.73, 1.98, 4.61,
       -0.5, 3.8, 0.16, -0.4, -1]
    df['tar_lin_full'] = tmp.multiply(coefs, axis=1).sum(axis=1) + np.random.normal(0, 11, size=entries)
    coef_dict['tar_lin_full'] = pd.DataFrame({'feat': tmp.columns, 'coef': coefs})
    
    # all the uncorrelated
    coefs = [3, 0.4, -0.01, 0.03, -2.4, 0.11, 0.1, 6, 2.36, 7.3, 0.75]
    df['tar_lin_unc'] = (tmp[[col for col in tmp.columns if col.startswith('unc_')]].multiply(coefs, axis=1).sum(axis=1) + 
                         np.random.normal(1, 0.4, size=entries))
    coef_dict['tar_lin_unc'] = pd.DataFrame({'feat': [col for col in tmp.columns if col.startswith('unc_')], 'coef': coefs})
    
    # all the correlated
    coefs = [3, 0.4, -0.01, 0.03, -2.4, 0.11, 0.1, -6, 2.36, -0.8]
    df['tar_lin_corr'] = (tmp[[col for col in tmp.columns if col.startswith('corr_')]].multiply(coefs, axis=1).sum(axis=1) + 
                          np.random.normal(4, 3, size=entries))
    coef_dict['tar_lin_corr'] = pd.DataFrame({'feat': [col for col in tmp.columns if col.startswith('corr_')], 'coef': coefs})
    
    # 3 variables
    coefs = [0.4, -0.01, 0.03]
    df['tar_lin_3'] = (tmp[['unc_normal_1', 'corr_normal_by_cat', 'corr_multinormal_mid_a']].multiply(coefs, axis=1).sum(axis=1) + 
                       np.random.normal(-0.7, 1, size=entries))
    coef_dict['tar_lin_3'] = pd.DataFrame({'feat': ['unc_normal_1', 'corr_normal_by_cat', 'corr_multinormal_mid_a'], 'coef': coefs})
    
    # 3 variables and interactions
    coefs = [0.4, -0.01, 0.03]
    df['tar_lin_3int'] = (tmp[['unc_normal_1', 'corr_normal_by_cat', 'corr_multinormal_mid_a']].multiply(coefs, axis=1).sum(axis=1) + 
                          0.6 * tmp['unc_normal_1'] * tmp['corr_normal_by_cat'] -
                          1.3 * tmp['corr_normal_by_cat'] * tmp['unc_normal_1'] * tmp['corr_normal_by_cat'] + 
                          np.random.normal(-0.7, 1, size=entries))
    coef_dict['tar_lin_3int'] = pd.DataFrame({'feat': ['unc_normal_1', 'corr_normal_by_cat', 'corr_multinormal_mid_a', 
                                                       'unc_normal_1__corr_normal_by_cat', 'all_3'], 
                                              'coef': coefs + [0.6, 1.3]})
    
    return df, coef_dict


def nonlinear_targets(data, tmp):
    df = data.copy()
    np.random.seed(23)
    entries = df.shape[0]
    
    coef_dict = {}
    
    # all
    coefs = [3, 0.4, 0.01, 0.03, 2.4, 0.11, 0.1, 6, 2.36, 7.3, 0.75, 0.69, 9.47, 0.73, 1.98, 4.61,
       0.5, 3.8, 0.16, 0.4, 1]
    df['tar_nonlin_full'] = 0.5*np.expm1(-tmp.multiply(coefs, axis=1).sum(axis=1) / 100) + np.random.normal(2, 2, size=entries)
    coef_dict['tar_nonlin_full'] = pd.DataFrame({'feat': tmp.columns, 'coef': coefs})
    
    # all the uncorrelated
    coefs = [3, 0.4, -0.01, 0.03, -2.4, 0.11, 0.1, 6, 2.36, 7.3, 0.75]
    df['tar_nonlin_unc'] = (np.expm1(tmp[[col for col in tmp.columns 
                                         if col.startswith('unc_')]].multiply(coefs, axis=1).sum(axis=1) / 20) + 
                            np.random.normal(0, 10, size=entries))
    coef_dict['tar_nonlin_unc'] = pd.DataFrame({'feat': [col for col in tmp.columns if col.startswith('unc_')], 'coef': coefs})
    
    # all the correlated
    coefs = [3, 0.4, -0.01, 0.03, -2.4, 0.11, 0.1, -6, 2.36, -0.8]
    df['tar_nonlin_corr'] = (np.expm1(tmp[[col for col in tmp.columns 
                                          if col.startswith('corr_')]].multiply(coefs, axis=1).sum(axis=1) / 20) + 
                             np.random.normal(4, 2, size=entries))
    coef_dict['tar_nonlin_corr'] = pd.DataFrame({'feat': [col for col in tmp.columns if col.startswith('corr_')], 'coef': coefs})
    
    # 3 variables
    df['tar_nonlin_3'] = (0.04*tmp['unc_ordinal']**2 - 0.09*tmp['corr_normal_by_cat'] + 0.8*np.tanh(tmp['corr_multinormal_mid_a']) + 
                          np.random.normal(-0.7, 10, size=entries))
    coef_dict['tar_nonlin_3'] = pd.DataFrame({'feat': ['unc_normal_1_squared', 'corr_normal_by_cat', 'corr_multinormal_mid_a_tanh'], 
                                           'coef': [0.04, -0.09, 0.8]})
    
    # 3 variables and interactions
    df['tar_nonlin_3int'] = (0.04*tmp['unc_ordinal']**2 - 0.09*tmp['corr_normal_by_cat'] - 0.8*np.tanh(tmp['corr_multinormal_mid_a']) + 
                          0.6 * tmp['unc_normal_1'] * tmp['corr_normal_by_cat'] -
                          0.3 * tmp['corr_normal_by_cat'] * tmp['unc_normal_1'] * tmp['corr_normal_by_cat'] + 
                          np.random.normal(-0.7, 10, size=entries))
    coef_dict['tar_nonlin_3int'] = pd.DataFrame({'feat': ['unc_normal_1_squared', 'corr_normal_by_cat', 'corr_multinormal_mid_a_tanh', 
                                                          'unc_normal_1__corr_normal_by_cat', 'all_3'], 
                                           'coef': [0.04, -0.09, 0.8, 0.6, -0.3]})
    
    return df, coef_dict


def make_targets(data):
    tmp = data.copy()
    tmp['unc_categories_100'] = tmp.unc_categories_100.astype('category').cat.codes.astype(int)
    tmp = pd.get_dummies(tmp, drop_first=True)
    
    df, coef_lin = linear_targets(data, tmp)
    df, coef_nonlin = nonlinear_targets(df, tmp)

    return df, coef_lin, coef_nonlin


def dirtify(data):
    # do not add missing values to the targets
    df = data[[col for col in data.columns if 'tar_' not in col]].copy()
    n_entries = df.shape[0]
    np.random.seed(23)
    
    # adding random noise to some columns
    df['unc_normal_1'] += np.random.normal(loc=0, scale=15, size=n_entries) / 10
    df['corr_normal_by_cat'] += np.random.normal(loc=-0.5, scale=5, size=n_entries) / 10
    df['unc_skewed_neg'] += np.random.normal(loc=0, scale=50, size=n_entries) / 10
    
    
    # make outliers
    random_entries = np.random.random(df.shape[0])<0.00001
    df['unc_normal_2'] = df['unc_normal_2'].mask(random_entries).fillna(df['unc_normal_2'] + 20)
    random_entries = np.random.random(df.shape[0])<0.00001
    df['corr_multinormal_mid_a'] = df['corr_multinormal_mid_a'].mask(random_entries).fillna(df['corr_multinormal_mid_a'] - 54)
    
    # adding 0.5% missing values
    missing = np.random.random(df.shape)<0.005
    df = df.mask(missing)
    
    # putting the targets back
    df = df.join(data[[col for col in data.columns if 'tar_' in col]])
    
    return df
