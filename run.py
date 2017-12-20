import numpy as np
import pandas as pd

from gather_data import get_split, get_tidy
import plotting
import modeling


x_train_tidy = get_tidy('train')
x_test_tidy = get_tidy('test')

y_train = get_split('y_train')
y_test = get_split('y_test')

#   Test set could randomly not have certain dummy variables
#   Ensure we only work with the intersection of columns
shared_columns = list(set(x_train_tidy.columns).intersection(set(x_test_tidy.columns)).intersection(modeling.COLS_TO_MODEL))

x_train_thin = x_train_tidy[shared_columns]
x_test_thin = x_test_tidy[shared_columns]



#--------------------------------------------------
#   Prepare model and train on data
#--------------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score

model = GradientBoostingClassifier()


z1 = modeling.cross_validate(model, x_train_thin, y_train, 'n_estimators', range(100, 101))
z1['param'] = 'baseline'

z2 = modeling.cross_validate(model, x_train_thin, y_train, 'n_estimators', range(50, 400, 50))
z3 = modeling.cross_validate(model, x_train_thin, y_train, 'learning_rate', np.arange(5, 21, 2.5) / 100)
z4 = modeling.cross_validate(model, x_train_thin, y_train, 'subsample', np.arange(7, 11) / 10)
z5 = modeling.cross_validate(model, x_train_thin, y_train, 'max_features', range(5, 13))


frames = [z1, 
          z2[z2.index==z2['mean'].idxmax()],
          z3[z3.index==z3['mean'].idxmax()],
          z4[z4.index==z4['mean'].idxmax()],
          z5[z5.index==z5['mean'].idxmax()]
          ]

pd.concat(frames)

        
#z2[z2.index==z2['mean'].idxmax()]





for s in range(4, 12):
    
    print(s)

    model = GradientBoostingClassifier(n_estimators = 400,
                                       learning_rate = .005,
                                       subsample = .85,
                                       max_features = 8,
                                       random_state = 125)
    
    scores = cross_val_score(model, x_train_thin, y_train, cv = 5)
    
    print(scores, scores.mean())
    










model = GradientBoostingClassifier(n_estimators = 400,
                                   subsample = .85,
                                   learning_rate = .005)

model.fit(x_train_thin, y_train)

y_train_pred = model.predict(x_train_thin)
y_test_pred = model.predict(x_test_thin)



#z = modeling.backward_feature_elimination(x_train_thin, y_train, model)


#--------------------------------------------------
#   Generate metrics to evaluate model performance
#--------------------------------------------------
print(classification_report (y_train,
                             y_train_pred,
                             target_names = ['Survived', 'Died']))

print(model.score(x_train_thin, y_train))


print(classification_report (y_test,
                             y_test_pred,
                             target_names = ['Survived', 'Died']))

print(model.score(x_test_thin, y_test))

plotting.show_roc(y_train, y_train_pred)
plotting.show_roc(y_test, y_test_pred)








#--------------------------------------------------------------------
#   Run test data through model and return results for submitting
#--------------------------------------------------------------------
x_eval = get_tidy('eval')

eval_shared_columns = sorted(list(set(x_eval.columns).intersection(modeling.COLS_TO_MODEL)))

X = x_eval[eval_shared_columns]

y_eval = model.predict(X)


"""
res = zip(x_eval.passengerid, y_eval)

import csv

csvfile = gather_data.FOLDERS['clean'] + "backward_gbm.csv"

with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['PassengerId', 'survived'])
    for line in res:
        id = line[0]
        value = line[1]
        writer.writerow([id, value])
"""        

#--------------------------------------------------------------------
#   Output data to be used by neural network. I need:
#   1) X_train, X_test and X_eval all matching columns
#   2) y_train, y_test
#--------------------------------------------------------------------
#   Just to see how neural network handles more data
#x_train_tidy = gather_data.tidy_data('all')
#y_train = x_train_tidy.survived



eval_shared_columns = set(x_eval.columns).intersection(modeling.COLS_TO_MODEL).intersection(set(x_train_tidy.columns))

np.savez(gather_data.FOLDERS['clean'] + 'titanic',
         x_train = x_train_tidy[list(eval_shared_columns)],
         x_test = x_test_tidy[list(eval_shared_columns)],
         x_eval = x_eval[list(eval_shared_columns)],
         y_train = y_train,
         y_test = y_test)




