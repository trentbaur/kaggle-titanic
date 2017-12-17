import statsmodels.formula.api as sm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import pdb

import gather_data


COLS_TO_MODEL = ['pclass', 
                 #'pclass_1', 'pclass_2', 'pclass_3', 
                 'age',
                 #'age_0_12', 'age_12_18', 'age_18_25', 'age_25_40', 'age_40_60', 'age_60_75',
                 'sibsp',
                 #'sibsp_0', 'sibsp_1', 'sibsp_2', 'sibsp_3', 'sibsp_4', 'sibsp_5', 'sibsp_8',
                 'parch',
                 #'parch_0', 'parch_1', 'parch_2', 'parch_3', 'parch_4', 'parch_5', 'parch_6', 
                 'fam_size',
                 #'fare',
                 'fare_log',
                 'is_alone',
                 'has_cabin',
                 #'embarked',
                 'embarked_c', 'embarked_q', 'embarked_s',
                 #'cabin_floor',
                 'cabin_floor_A', 'cabin_floor_B', 'cabin_floor_C', 'cabin_floor_D', 'cabin_floor_E', 'cabin_floor_F', 'cabin_floor_G', 'cabin_floor_T',
                 #  Gender dummy variables, placed last to avoid comma-commenting issues
                 #'sex_female',
                 'sex_male']

#COLS_TO_MODEL = ['fam_size', 'pclass_3', 'cabin_floor_A', 'pclass_1', 'cabin_floor_C', 'parch', 'sibsp', 'age_12_18', 'sex_male', 'embarked_c', 'parch_1', 'fare_log', 'cabin_floor_E', 'cabin_floor_B', 'has_cabin', 'embarked_s', 'pclass', 'age', 'age_18_25', 'cabin_floor_D']

def build_model_age():
    
    if 'predict_age' not in globals():
        
        x_train = gather_data.get_data('x_train')
    
        formula = 'age ~ sex + pclass'
    
        model_age = sm.ols(formula, data = x_train).fit()
        
        globals()['predict_age'] = model_age
        
    return globals()['predict_age']

#   build_model_age()

def model_age(df):
    
    model = build_model_age()

    age_pred = model.predict(df[df.age.isnull()])
    
    return np.ceil(age_pred)
    
#   model_age(gather_data.get_data())


#-------------------------
#   Feature Selection
#-------------------------
def backward_feature_elimination(x, y, model):
    
    results = []
    
    #   Initialize storage object with full set of columns
    features = x.columns

    while len(features) > 4:
        print('Feature Count: {}'.format(len(features)))
        
        #   Run model
        model.fit(x[features], y)

        score = f1_score(y, model.predict(x[features]), average="macro")
        
        #   Retrieve feature importance and store values
        values = dict(zip(features, model.feature_importances_))
        values['score'] = score
        results.append(values)
        
        #   Eliminate feature
        low_import = min(values.values())
        print('Lowest Importance: ()'.format(low_import))
        #pdb.set_trace()
        features = [k for k, v in values.items() if (v > low_import) & (k != 'score')]
   
    return results

"""
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators = 500,
                                   learning_rate = .05)
z = backward_feature_elimination(x_train_thin, y_train, model)

pd.DataFrame(z)['score']
pd.DataFrame(z)['score'].max()




from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 500,
                               max_features = 4,
                               random_state = 0)
z = backward_feature_elimination(x_train_thin, y_train, model)

pd.DataFrame(z)['score']
pd.DataFrame(z)['score'].max()


    values = dict(zip(x_train_thin.columns, model.feature_importances_))
    score = model.score(x_train_thin, y_train)
    values['score'] = score
"""