import gather_data
from sklearn.model_selection import train_test_split

#----------------------------
#   Gather training data 
#----------------------------
train = gather_data.load_data('train.csv')



#------------------------------
#   Summarize / Explore Data
#------------------------------
#   train.groupby('Survived').count()
'''
#   Explore records with no age value
train[train.age.isnull()]

name_stub = 'Master'

train[train.Name.str.contains(name_stub)]

import matplotlib.pyplot as plt
plt.hist(train[(train.Name.str.contains('Mrs')) & (train.Age.notnull())].Age, bins=10)

plt.hist(train[(train.age > 0)].fare, bins=10)


'''



#------------------------------
#   Prepare / Tidy data
#------------------------------
y_data = train.Survived
train.drop(labels = ['Survived'], axis = 1, inplace = True)

train = gather_data.tidy_data(train)

cols_to_model = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'female', 'male', 'c', 'q', 's']

X_train, X_dev, y_train, y_dev = train_test_split(train[cols_to_model],
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
'''

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 1000,
                               max_features = 3,
                               random_state = 0)

model.fit(X_train, y_train)

#--------------------------------------------------
#   Generate metrics to evaluate model performance
#--------------------------------------------------
from sklearn.metrics import classification_report

print(classification_report (y_dev,
                             model.predict(X_dev),
                             target_names = ['Survived', 'Died']))







#--------------------------------------------------------------------
#   Run test data through model and return results for submitting
#--------------------------------------------------------------------
test = gather_data.load_data('test.csv')
#   test[test.fare.isnull()]

x_test = gather_data.tidy_data(test)
x_test = x_test[cols_to_model]

x_test[x_test.age.isnull()]
x_test[x_test.fare.isnull()]

y_pred = model.predict(x_test)

res = zip(test.passengerid, y_pred)

import csv

csvfile = "randomforest.csv"

with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['PassengerId', 'Survived'])
    for line in res:
        id = line[0]
        value = line[1]
        writer.writerow([id, value])
        

