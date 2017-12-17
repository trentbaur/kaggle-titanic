import gather_data, plotting, modeling

#------------------------------------------------------
#   Load raw data and split between training and dev
#------------------------------------------------------
train_raw = gather_data.load_data('train')

gather_data.split_data(125)


#-------------------------------------------------------------------
#   Create tidy dataset that will be used for analysis
#-------------------------------------------------------------------
x_train_thin = gather_data.tidy_data('train')[modeling.COLS_TO_MODEL]
x_test_thin = gather_data.tidy_data('test')[modeling.COLS_TO_MODEL]


#--------------------------------------------------
#   Prepare model and train on data
#--------------------------------------------------
'''
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C = 1)

model.fit(X_train, y_train)



from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()

model.fit(X_train, y_train)
'''

y_train = gather_data.get_data('y_train')
y_test = gather_data.get_data('y_test')

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 10,
                               max_features = 3,
                               random_state = 0)

model.fit(x_train_thin, y_train)


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
x_eval = x_eval[modeling.COLS_TO_MODEL]

y_eval = model.predict(x_eval)


"""
res = zip(test.passengerid, y_pred)

import csv

csvfile = "gbrt.csv"

with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['PassengerId', 'survived'])
    for line in res:
        id = line[0]
        value = line[1]
        writer.writerow([id, value])
"""        

