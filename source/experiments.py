__author__ = 'lucabasa'
__version__ = '1.3.0'
__status__ = 'development'

import source.hyperplots as hyp
from source.report import plot_predictions, make_results, store_results
from source.utility import cv_score, grid_search
import source.transf_univ as df_p
from source.clean import general_cleaner, drop_columns  # for houseprice
from source.transf_category import make_ordinal, recode_cat  # for houseprice
from source.transf_numeric import tr_numeric  # for houseprice
from sklearn.pipeline import Pipeline
import random
import pandas as pd
import numpy as np

import warnings


def _import_generated_data(target_name):
    df = pd.read_csv('data/simulated/clean.csv')
    coefficients = pd.read_pickle('data/simulated/coefficients.pkl')
    coef_names = list(coefficients[target_name].feat.values)
    df['target'] = df[target_name]
    del df[target_name]
    coefs_file = None
    
    return df, coef_names, coefs_file


def _import_sklearn_data(data_name):
    df = pd.read_csv(data_name)
    coefs_file = data_name.split('.csv')[0] + '__coefficients.csv'
    
    true_coefficients = pd.read_csv(coefs_file)
    coef_names = list(true_coefficients.variable.unique())
    
    return df, coef_names, coefs_file


def import_hp():
    df = pd.read_csv('data/hp/train.csv')
    df['target'] = np.log1p(df.SalePrice)
    del df['SalePrice']
    df = df[df.GrLivArea < 4500].reset_index(drop=True)
    return df


def hp_model(model):
    numeric_pipe = Pipeline([('fs', df_p.feat_sel('numeric')),
                         ('imputer', df_p.df_imputer(strategy='median')),
                         ('transf', tr_numeric())])

    cat_pipe = Pipeline([('fs', df_p.feat_sel('category')),
                         ('imputer', df_p.df_imputer(strategy='most_frequent')), 
                         ('ord', make_ordinal(['BsmtQual', 'ExterQual', 'HeatingQC'])), 
                         ('recode', recode_cat()), 
                         ('dummies', df_p.dummify())])

    processing_pipe = df_p.FeatureUnion_df(transformer_list=[('cat_pipe', cat_pipe),
                                                     ('num_pipe', numeric_pipe)])
    
    pipe = [('gen_cl', general_cleaner()),
            ('processing', processing_pipe),
            ('scl', df_p.df_scaler(method='robust')),
            ('dropper', drop_columns())] + [model]
    
    model_pipe = Pipeline(pipe)
    
    return model_pipe


def make_exp(model, kfolds, hp=False, data_name=None, target_name=None, features='all',
            sample=False, store=False, coefs=True, store_name=None, parameters=None, model_name=None):
    '''
    This is a wrapper to perform an experiment on the data generated by sklearn or by the generate_data module
    
    model: pipeline of at least 2 steps or tuple (name, algorithm). If hp=True, it must be a tuple with the final
        step of the pipeline.
    kfolds: KFold instance to use
    hp: if True, it imports the data and build the pipeline for HousePrice
    data_name: if used, indicates data generated via sklearn. If None, it will use the data generated by me
    target_name: Mandatory if data_name is None, it indicates which target variable we are considering
    features: string for feature selection
    sample: integer to select only a sample of the data
    store: boolean to store the result on a csv
    coefs: boolean, it true it plots the coefficients too
    store_name: path to file where the summary of the results is stored
    parameters: dictionary to be stored
    model_name: name of the model to be stored
    
    '''
    random.seed(666)
    if hp:
        df = _import_hp()
        target_name = 'HP'
        coef_names = ['']
        features = 'all'
        coefs = False
        model = hp_model(model)
    elif data_name is None:
        df, coef_names, coefs_file = _import_generated_data(target_name)
    else:
        df, coef_names, coefs_file = _import_sklearn_data(data_name)
        target_name = data_name.split('/')[2].split('.csv')[0]
    if sample:
        df = df.sample(sample)

    target = df['target']
    
    if not hp:
        df = pd.get_dummies(df, drop_first=True)

    df_train = df.drop('target', axis=1)
    
    df_train = select_features(features, df_train, coef_names)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The dummies in this set do not match the ones in the train set, we corrected the issue.")
        oof, coefs_est = cv_score(df_train, target, kfolds, model, imp_coef=True)

    plot_predictions(df_train, target, oof, feature=None, hue=None, legend=False, savename=False)
    
    if hp:
        hyp.plot_coef_est(coefs_est)
    
    if coefs:
        hyp.plot_coefficients(target_name, coefs_est, coefs_real=coefs_file)
    
    if store:
        store_results(store_name, 
                      label=target, prediction=oof, model=model_name, 
                      parameters=parameters, 
                      target_name=target_name, variables=features, instances=df_train.shape[0], verbose=True)
    else:
        res = make_results(label=target, prediction=oof, model='not_relevant', 
                           parameters='not_relevant', 
                           target_name=target_name, variables=features, instances=df_train.shape[0], verbose=True)


def learning_curve(model, kfolds, data_name=None, target_name=None, 
                   features='all', sample=None, scoring='neg_mean_absolute_error'):
    '''
    Wrapper around hyp.plot_learning_curve to facilitate feature selection and instances selection
    
    model: anything with a fit and predict method (pipeline or not)
    data_name: if used, indicates data generated via sklearn. If None, it will use the data generated by me
    target_name: Mandatory if data_name is None, it indicates which target variable we are considering
    features: string for feature selection
    sample: if not None, selects a subsample of instances
    scoring: scoring function to evaluate the curves
    '''
    random.seed(666)
    if data_name is None:
        df, coef_names, coefs_file = _import_generated_data(target_name)
        title = target_name
    else:
        df, coef_names, coefs_file = _import_sklearn_data(data_name)
        title = data_name.split('/')[2].split('.csv')[0]
    if sample:
        df = df.sample(sample)

    target = df['target']
    df = pd.get_dummies(df, drop_first=True)

    df_train = df.drop('target', axis=1)
    
    df_train = select_features(features, df_train, coef_names)

    hyp.plot_learning_curve(model, title, df_train, target, scoring=scoring, cv=kfolds, n_jobs=-1)

        
def select_features(features, data, coef_names):
    random.seed(4561)
    if features == 'all':
        return data[[col for col in data.columns if 'tar_' not in col]]
    elif features == 'exact':
        return data[coef_names]
    elif features == 'exact-10':
        n_used = len(coef_names)
        to_use = random.sample(coef_names, int(n_used*0.9))
        return data[to_use]
    elif features == 'unobserved':
        n_used = len(coef_names)
        to_drop = random.sample(coef_names, int(n_used*0.1))
        return data[[col for col in data.columns if 'tar_' not in col]].drop(to_drop, axis=1)
    else:
        raise KeyError('Wrong feature selection provided. Use all, exact, exact-10, or unobserved')
        
        