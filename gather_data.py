import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import modeling


FOLDERS = {'raw': 'data_raw/',
           'clean': 'data_clean/'}

def load_data(data_type = 'train'):
    if data_type + '_raw' not in globals():
        print('Loading ' + data_type + '_raw')
        
        df = pd.read_csv(FOLDERS['raw'] + data_type + '.csv')
        
        df.columns = [col.lower() for col in df.columns]

        globals()[data_type + '_raw'] = df
        
    return globals()[data_type + '_raw']


def split_data(p_random = 125):
    train_raw = load_data()
    
    x_train, x_test, y_train, y_test = train_test_split(train_raw,
                                                        train_raw.survived,
                                                        random_state = p_random)
    
    globals()['x_train'] = x_train
    globals()['x_test'] = x_test
    globals()['y_train'] = y_train
    globals()['y_test'] = y_test
    
    return

#   split_data()


def get_data(p_name = 'x_train'):
    
    if p_name == 'x_eval':
        return load_data('test')
    
    elif p_name not in globals():
        split_data()
    
    return globals()[p_name]

#   get_data()


#----------------------------------------------------------
#   Tidy Data Function
#   This must be suitable for both training and test data
#       and the final test data
#----------------------------------------------------------
def tidy_data(p_type = 'train'):

    if p_type + '_tidy' not in globals():

        data = get_data('x_' + p_type).copy()
        
        print('Loading ' + p_type + '_tidy')
        
        #   Fill in missing age values based on model
        data.loc[data.age.isnull(), 'age'] = modeling.model_age(data)

        #   Place ages into bins    
        bins = [0, 12, 18, 25, 40, 60, 75]
       
        data['age_bin'] = pd.cut(data['age'], bins, labels = ['age_0_12', 'age_12_18', 'age_18_25', 'age_25_40', 'age_40_60', 'age_60_75'])
        enc_age = pd.get_dummies(data.age_bin)
        data = pd.concat([data, enc_age], axis = 1)
    
        data.embarked = data.embarked.str.lower()

        data['cabin_floor'] = data.cabin.str.replace('[0-9]| ', '').str.get(0).astype('category')

        for col in ['pclass', 'sex', 'sibsp', 'parch', 'embarked', 'cabin_floor']:
            encode = pd.get_dummies(data[col], prefix = col)
            data = pd.concat([data, encode], axis = 1)

        #   Create interaction variables
        data.loc[(data.fare.isnull()), 'fare'] = data.fare.describe()['50%']
        data['age*male'] = data.age * data.sex_male
        data['fare_log'] = np.log(data.fare)
        data['fam_size'] = data.sibsp + data.parch
        data['ticket_class'] = data.ticket.str.replace('[0-9]| ', '')

        #   Create interaction variables
        for col in ['pclass', 'sibsp', 'parch', 'embarked', 'fam_size', 'ticket_class']:
            data[col] = data[col].astype('category')
                
        globals()[p_type + '_tidy'] = data

    return globals()[p_type + '_tidy']

#   tidy_data().columns
#   tidy_data('test')
#   tidy_data('eval')
#   tidy_data().dtypes

