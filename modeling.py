import pandas as pd
import statsmodels.formula.api as sm

import gather_data

COLS_TO_MODEL = ['pclass', 
                 #'age*class',
                 'age',
                 'sibsp',
                 'parch',
                 'fam_size',
                 'fare',
                 #  Embarked dummy variables
                 'c', 'q', 's',
                 #  Cabin Floor dummy variables
                 #'A', 'B', 'C', 'D', 'E', 'F', 'G', 'T',
                 #'age_0_12', 'age_12_18', 'age_18_25', 'age_25_40', 'age_40_60', 'age_60_75'
                 #  Gender dummy variables, placed last to avoid comma-commenting issues
                 'female',
                 'male']


def logreg_sig(p_vars = 'age * male'):
    
    df = gather_data.load_combined_tidy()
    
    formula = 'survived ~ ' + p_vars
    
    results = sm.ols(formula, data = df).fit()
    
    return results.summary()

#   logreg_sig()

#   logreg_sig('pclass + fare')
    
def model_age():
    
    df = gather_data.load_data()
    
    formula = 'Age ~ Sex + Pclass'
    
    results = sm.ols(formula, data = df).fit()
    
    return results.summary()

#   model_age()


