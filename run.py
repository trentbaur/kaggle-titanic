import csv

from gather_data import get_split, get_tidy, COLS_TO_MODEL, FOLDERS
import plotting
import modeling

x_train_tidy = get_tidy('train')
x_test_tidy = get_tidy('test')

y_train = get_split('y_train')
y_test = get_split('y_test')

#   x_train_tidy COULD be missing a dummy column that test_tidy or eval_tidy have
#       but we can't consider those fields anyways so exclude them
#   test_tidy and eval_tidy will be padded with any missing columns that train_tidy has
shared_columns = sorted(list(set(x_train_tidy.columns).intersection(set(x_test_tidy.columns)).intersection(COLS_TO_MODEL)))

x_train_thin = x_train_tidy[shared_columns]
x_test_thin = x_test_tidy[shared_columns]


#--------------------------------------------------
#   Prepare model and train on data
#--------------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

model = GradientBoostingClassifier(n_estimators = 50,
                                   learning_rate = .025,
                                   subsample = .9,
                                   max_features = 6,
                                   max_depth = 7)

# model = SVC(C = 100000000, gamma = 1e-08)

# model= RandomForestClassifier(n_estimators = 90,
#                               max_features = 15)

#   modeling.backward_feature_elimination(model, x_train_thin, y_train)


model.fit(x_train_thin, y_train)

y_train_pred = model.predict(x_train_thin)
y_test_pred = model.predict(x_test_thin)


#--------------------------------------------------
#   Generate metrics to evaluate model performance
#--------------------------------------------------
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print(classification_report (y_train,
                             y_train_pred,
                             target_names = ['Survived', 'Died']))

print(accuracy_score(y_train, y_train_pred))

print(classification_report (y_test,
                             y_test_pred,
                             target_names = ['Survived', 'Died']))

print(accuracy_score(y_test, y_test_pred))

plotting.show_roc(y_train, y_train_pred)
plotting.show_roc(y_test, y_test_pred)


#--------------------------------------------------------------------
#   Run test data through model and return results for submitting
#--------------------------------------------------------------------
x_eval = get_tidy('eval')

eval_shared_columns = sorted(list(set(x_train_tidy.columns).intersection(set(x_eval.columns).intersection(COLS_TO_MODEL))))

X = x_eval[eval_shared_columns]

y_eval = model.predict(X)

y_eval.sum()
    

#----------------------------------------------
#   Save eval predictions for submission
#----------------------------------------------
res = zip(x_eval.passengerid, y_eval)

csvfile = FOLDERS['clean'] + "scaled.csv"

with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['PassengerId', 'survived'])
    for line in res:
        id = line[0]
        value = line[1]
        writer.writerow([id, value])


#--------------------------------------------------------------------
#   Output data to be used by neural network. I need:
#   1) X_train, X_test and X_eval all matching columns
#   2) y_train, y_test
#--------------------------------------------------------------------
#   Just to see how neural network handles more data
#x_train_tidy = gather_data.tidy_data('all')
#y_train = x_train_tidy.survived


"""
import numpy as np

eval_shared_columns = set(x_eval.columns).intersection(COLS_TO_MODEL).intersection(set(x_train_tidy.columns))

np.savez(gather_data.FOLDERS['clean'] + 'titanic',
         x_train = x_train_tidy[list(eval_shared_columns)],
         x_test = x_test_tidy[list(eval_shared_columns)],
         x_eval = x_eval[list(eval_shared_columns)],
         y_train = y_train,
         y_test = y_test)

"""


