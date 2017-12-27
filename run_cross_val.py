import numpy as np

from gather_data import get_split, get_tidy, COLS_TO_MODEL
import modeling

x_train_tidy = get_tidy('train')
x_test_tidy = get_tidy('test')

y_train = get_split('y_train')
y_test = get_split('y_test')

#   x_train_tidy COULD be missing a dummy column that test_tidy or eval_tidy
#       but we can't consider those fields anyways so exclude them
#   test_tidy and eval_tidy will be padded with any missing columns that train_tidy has
shared_columns = sorted(list(set(x_train_tidy.columns).intersection(set(x_test_tidy.columns)).intersection(COLS_TO_MODEL)))

x_train_thin = x_train_tidy[shared_columns]
x_test_thin = x_test_tidy[shared_columns]


#-------------------------------
#   Gradient Boosted Trees
#-------------------------------
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()

params = {'n_estimators': range(20, 150, 10),
          'learning_rate': np.arange(.02, .1, .01),
          'subsample': np.arange(7, 10.1, .5) / 10,
          'max_features': range(4, len(x_train_thin.columns)),
          'max_depth': range(2, 10)}

model = modeling.cross_all(model, x_train_thin, y_train, params)



#------------------------
#   Random Forests
#------------------------
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_jobs = -1)

params = {'n_estimators': range(20, 120, 10),
          'max_features': range(5, 16)}

model = modeling.cross_all(model, x_train_thin, y_train, params)


#-------------------------------
#   Support Vector Classifier
#-------------------------------
from sklearn.svm import SVC

model = SVC()

params = {'C': (.001, .01, .1, 1, 10, 100, 1000, 10000, 100000, 1000000),
          'gamma': (.000001, .00001, .0001, .001, .01, .1, 1, 10, 100)}

model = modeling.cross_all(model, x_train_thin, y_train, params)


#-------------------------------------------------------------
#   Use Backward Feature Elimination against resulting model
#-------------------------------------------------------------
modeling.backward_feature_elimination(model, x_train_thin, y_train)


#------------------------------------------------------------
#   Use scikit-learn's Grid Search Cross Validation for SVC
#       Takes a very long time to run
#------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
grid = GridSearchCV(SVC(), param_grid = param_grid, cv = 5, n_jobs = 4)
grid.fit(x_train_thin, y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

#The best parameters are {'C': 100000000.0, 'gamma': 1e-08} with a score of 0.82
