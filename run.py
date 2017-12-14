import gather_data


train = gather_data.load_data('train.csv')

train = train[train.Age.notnull()]
y_train = train.Survived
train.drop(['Survived'], axis=1, inplace=True)

x_train = gather_data.tidy_data(train)

#   x_train.head()


cols_to_include = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'female', 'male', 'C', 'Q', 'S']

x_train = x_train[cols_to_include]


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=10)

model.fit(x_train, y_train)

test = gather_data.load_data('test.csv')
test = test[test.Age.notnull() & test.Fare.notnull()]

x_test = gather_data.tidy_data(test)
x_test = x_test[cols_to_include]

model.predict(x_test)