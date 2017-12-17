import gather_data, plotting, modeling
import numpy as np

#------------------------------------------------------
#   Load raw data and split between training and dev
#------------------------------------------------------
train_raw = gather_data.load_data('train')

gather_data.split_data(125)


#-------------------------------------------------------------------
#   Create tidy dataset that will be used for analysis
#-------------------------------------------------------------------

x_train_tidy = gather_data.tidy_data('train')
x_test_tidy = gather_data.tidy_data('test')

y_train = gather_data.get_data('y_train')
y_test = gather_data.get_data('y_test')

shared_columns = list(set(x_test_tidy.columns).intersection(modeling.COLS_TO_MODEL))

x_train_thin = x_train_tidy[shared_columns]
x_test_thin = x_test_tidy[shared_columns]


#   Just to see how neural network handles more data
#x_train_tidy = gather_data.tidy_data('all')
#y_train = x_train_tidy.survived

#--------------------------------------------------
#   Prepare model and train on data
#--------------------------------------------------
'''
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C = 1)

model.fit(x_train_thin, y_train)



from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators = 500,
                                   learning_rate = .05)

model.fit(x_train_thin, y_train)


from sklearn.linear_model import RidgeClassifier

model = RidgeClassifier(alpha = 1, max_iter=100000)

model.fit(x_train_thin, y_train)


from sklearn.linear_model import Lasso

model = Lasso(alpha = 1, max_iter=100000)

model.fit(x_train_thin, y_train)

'''

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 200,
                               max_features = 3,
                               random_state = 0)

model.fit(x_train_thin, y_train)


sorted(list(zip(shared_columns, model.feature_importances_)), key=lambda x: x[0])


#--------------------------------------------------
#   Generate metrics to evaluate model performance
#--------------------------------------------------
from sklearn.metrics import classification_report

print(classification_report (y_train,
                             model.predict(x_train_thin),
                             target_names = ['Survived', 'Died']))

print(classification_report (y_test,
                             model.predict(x_test_thin),
                             target_names = ['Survived', 'Died']))

plotting.show_roc(y_train, model.predict(x_train_thin))
plotting.show_roc(y_test, model.predict(x_test_thin))


#--------------------------------------------------------------------
#   Run test data through model and return results for submitting
#--------------------------------------------------------------------
x_eval = gather_data.tidy_data('eval')

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
eval_shared_columns = set(x_eval.columns).intersection(modeling.COLS_TO_MODEL).intersection(set(x_train_tidy.columns))

np.savez(gather_data.FOLDERS['clean'] + 'titanic',
         x_train = x_train_tidy[list(eval_shared_columns)],
         x_test = x_test_tidy[list(eval_shared_columns)],
         x_eval = x_eval[list(eval_shared_columns)],
         y_train = y_train,
         y_test = y_test)




