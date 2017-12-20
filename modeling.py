import pandas as pd
import numpy as np

from sklearn.metrics import f1_score


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


import pdb

#-------------------------
#   Cross Validate
#-------------------------
from sklearn.model_selection import cross_val_score

def cross_validate(model, x, y, param = 'n_estimators', n_range = range(1,2), metric = 'roc_auc'):
    
    all_scores = []
    
    #pdb.set_trace()
    
    for val in n_range:
        
        print(val)
        
        setattr(model, param, val)
        
        scores = cross_val_score(model, x, y, cv = 5, scoring = metric)
        
        all_scores.append((param, val, scores.mean(), scores.std(), scores.min(), scores.max()))
    
    return pd.DataFrame(all_scores, columns = ['param', 'value', 'mean', 'std', 'min', 'max'])
        
        


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
    
    
    
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report


from sklearn.model_selection import KFold

kf = KFold(n_splits= 5, random_state=1)

for train_index, test_index in kf.split(x_train_thin):
    print('Train: ', train_index, 'Test: ', test_index)
    x_train_kf, x_test_kf = x_train_thin[train_index], x_train_thin[test_index]
    y_train_kf, y_test_kf = y_train[test_index], y_test[test_index]
    
    





model = LogisticRegression(C = 1)

model = GradientBoostingClassifier()

model = GradientBoostingClassifier(n_estimators = 500,
                                   learning_rate = .05)

model = RidgeClassifier(alpha = 1, max_iter=100000)

model = Lasso(alpha = 1, max_iter=100000)

model = RandomForestClassifier(n_estimators = 200,
                               max_features = 3,
                               random_state = 0)
"""