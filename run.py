import gather_data, explore, plotting, modeling
from sklearn.model_selection import train_test_split


#------------------------------
#   Prepare / Tidy data
#------------------------------
train_raw = gather_data.load_data('train')
y_data = gather_data.load_y()


train_tidy = gather_data.tidy_data('train')


x_train, x_dev, y_train, y_dev = train_test_split(train_tidy[modeling.COLS_TO_MODEL],
                                                  y_data,
                                                  random_state = 0)

#   X_train.head()



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

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 10,
                               max_features = 3,
                               random_state = 0)

model.fit(x_train, y_train)


#--------------------------------------------------
#   Generate metrics to evaluate model performance
#--------------------------------------------------
from sklearn.metrics import classification_report

print(classification_report (y_train,
                             model.predict(x_train),
                             target_names = ['Survived', 'Died']))

print(classification_report (y_dev,
                             model.predict(x_dev),
                             target_names = ['Survived', 'Died']))

plotting.show_roc(y_train, model.predict(x_train))
plotting.show_roc(y_dev, model.predict(x_dev))


#--------------------------------------------------------------------
#   Run test data through model and return results for submitting
#--------------------------------------------------------------------
test = gather_data.load_data('test')

x_test = gather_data.tidy_data('test')
x_test = x_test[modeling.COLS_TO_MODEL]

x_test[x_test.age.isnull()]
x_test[x_test.fare.isnull()]

y_pred = model.predict(x_test)


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

