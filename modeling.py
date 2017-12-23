import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

import plotting


#-------------------------
#   Feature Selection
#-------------------------
def backward_feature_elimination(model, x, y):
    
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

        features = [k for k, v in values.items() if (v > low_import) & (k != 'score')]
   
    return results


#-------------------------
#   Cross Validate
#-------------------------

def cross_validate(model, x, y, param = 'n_estimators', n_range = range(1,2), metric = 'accuracy'):
    
    all_scores = []
    
    for val in n_range:
        
        print(val)
        
        setattr(model, param, val)
        
        scores = cross_val_score(model, x, y, cv = 5, scoring = metric)
        
        all_scores.append((param, val, scores.mean(), scores.std(), scores.min(), scores.max()))
    
    return pd.DataFrame(all_scores, columns = ['param', 'value', 'mean', 'std', 'min', 'max'])
        
        
def cross_all(model, x, y, params, metric = 'accuracy'):
    
    best = pd.DataFrame()

    baseline = cross_validate(model, x, y, 'n_estimators', range(100, 101))
    baseline['param'] = 'baseline'

    best = best.append(baseline)

    cv_results = {}

    for k in params:
        results = cross_validate(model, x, y, k, params[k])
        
        cv_results[k] = results
        
        best = best.append(results[results.index==results['mean'].idxmax()])
    
    #   Display all four params
    plotting.show_barplot_all(cv_results)

    #   Select best param and set in model. Remove from params
    print(best)
    
    return best
"""


    
    
    
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier




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