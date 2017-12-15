import gather_data
import plotting
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

cols_to_model = ['pclass', 'ageIclass', 'age', 'sibsp', 'parch', 'fam_size', 'fare', 'female', 'male', 'c', 'q', 's']#, 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']
                 #'age_0_12', 'age_12_18', 'age_18_25', 'age_25_40', 'age_40_60', 'age_60_75']
#cols_to_model = ['pclass', 'ageIclass', 'age', 'sibsp', 'parch', 'fam_size', 'fare', 'female', 'male', 'c', 'q', 's', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'T']

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



from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()

model.fit(X_train, y_train)
'''

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 10,
                               max_features = 3,
                               random_state = 0)

model.fit(X_train, y_train)


#--------------------------------------------------
#   Generate metrics to evaluate model performance
#--------------------------------------------------
from sklearn.metrics import classification_report, roc_curve

print(classification_report (y_train,
                             model.predict(X_train),
                             target_names = ['Survived', 'Died']))

print(classification_report (y_dev,
                             model.predict(X_dev),
                             target_names = ['Survived', 'Died']))

plotting.show_roc(y_train, model.predict(X_train))
plotting.show_roc(y_dev, model.predict(X_dev))


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

csvfile = "gbrt.csv"

with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['PassengerId', 'Survived'])
    for line in res:
        id = line[0]
        value = line[1]
        writer.writerow([id, value])
        

