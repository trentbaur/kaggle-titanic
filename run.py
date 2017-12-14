import gather_data
from sklearn.model_selection import train_test_split

#----------------------------
#   Gather training data 
#----------------------------
train = gather_data.load_data('train.csv')
#train.dropna(subset = ['Age'], inplace = True)



#------------------------------
#   Summarize / Explore Data
#------------------------------
#   train.groupby('Survived').count()



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
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C = 1)

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
test.dropna()
test = test[test.Age.notnull() & test.Fare.notnull()]

x_test = gather_data.tidy_data(test)
x_test = x_test[cols_to_model]

model.predict(x_test)

