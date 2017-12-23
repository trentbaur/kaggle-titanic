import numpy as np
import pandas as pd

from gather_data import get_split, get_tidy, COLS_TO_MODEL
import plotting
import modeling

x_train_tidy = get_tidy('train')
x_test_tidy = get_tidy('test')

y_train = get_split('y_train')
y_test = get_split('y_test')

#   x_train_tidy COULD be missing a dummy column that test_tidy or eval_tidy
#       but we can't consider those fields anyways so exclude them
#   test_tidy and eval_tidy will be padded with any missing columns that train_tidy has
#x_train_thin = x_train_tidy[[x for x in COLS_TO_MODEL if x in x_train_tidy.columns]]
#x_test_thin = x_test_tidy[x_train_thin.columns.values]

shared_columns = sorted(list(set(x_train_tidy.columns).intersection(set(x_test_tidy.columns)).intersection(COLS_TO_MODEL)))

x_train_thin = x_train_tidy[shared_columns]
x_test_thin = x_test_tidy[shared_columns]


#--------------------------------------------------
#   Prepare model and train on data
#--------------------------------------------------
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()


params = {'n_estimators': range(25, 200, 25),
          'learning_rate': np.arange(.025, .2, .025),
          'subsample': np.arange(7, 10.1, .5) / 10,
          'max_features': range(5, 16)}


x = modeling.cross_all(model, x_train_thin, y_train, params)

#-

model = GradientBoostingClassifier(n_estimators = 50)

params = {'learning_rate': np.arange(.025, .2, .025),
          'subsample': np.arange(7, 10.1, .5) / 10,
          'max_features': range(5, 16)}


modeling.cross_all(model, x_train_thin, y_train, params)

#-

model = GradientBoostingClassifier(n_estimators = 50,
                                   learning_rate = .025)

params = {'subsample': np.arange(7, 10.1, .5) / 10,
          'max_features': range(5, 16)}


modeling.cross_all(model, x_train_thin, y_train, params)

#-

model = GradientBoostingClassifier(n_estimators = 50,
                                   learning_rate = .025,
                                   max_features = 10)

params = {'subsample': np.arange(7, 10.1, .5) / 10}


modeling.cross_all(model, x_train_thin, y_train, params)

#--------------

model = GradientBoostingClassifier(n_estimators = 50,
                                   learning_rate = .025,
                                   subsample = .75,
                                   max_features = 10)


