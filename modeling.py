import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import plotting


#-------------------------
#   Feature Selection
#-------------------------
def backward_feature_elimination(model, x, y):
    
    results = []
    
    #   Initialize storage object with full set of columns
    features = x.columns

    while len(features) > getattr(model, 'max_features'):
        print('Feature Count: {}'.format(len(features)))
        
        #   Run model
        model.fit(x[features], y)

        score = accuracy_score(y, model.predict(x[features]))
        
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
        
        print(param + ': ' + str(val))
        
        setattr(model, param, val)
        
        scores = cross_val_score(model, x, y, cv = 5, scoring = metric)
        
        all_scores.append((param, val, scores.mean(), scores.std(), scores.min(), scores.max()))
    
    return pd.DataFrame(all_scores, columns = ['param', 'value', 'mean', 'std', 'min', 'max'])


def cross_all(model, x, y, params, metric = 'accuracy'):
    
    while len(params) > 0:
        
        cv_best = pd.DataFrame()
        cv_results = {}
    
        for k in params:
            results = cross_validate(model, x, y, k, params[k])
            
            cv_results[k] = results
            
            cv_best = cv_best.append(results[results.index==results['mean'].idxmax()])
        
        #   Display all four params
        #plotting.show_barplot_all(cv_results)
        
        cv_best.reset_index(inplace = True)
        
        #   Select best param and set in model. Remove from params
        param, value = cv_best.loc[cv_best.index == cv_best['mean'].idxmax(), ['param', 'value']].values.ravel()
        
        if param in ('max_features', 'n_estimators'):
            value = int(value)
            
        print(cv_best)
        print(param)
        print(value)
        
        setattr(model, param, value)
        
        del params[param]
        
    return model

