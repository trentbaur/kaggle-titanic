import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import statsmodels.formula.api as sm

FOLDERS = {'raw': 'data_raw/',
           'clean': 'data_clean/'}

def get_raw(data_type = 'train'):
    
    filename = data_type + '_raw'
    
    if filename not in globals():
        
        print('Loading ' + filename)
        
        df = pd.read_csv(FOLDERS['raw'] + data_type + '.csv')
        
        df.columns = [col.lower() for col in df.columns]
        
        #   Placing here as a quick a dirty way of avoiding log(0)
        #   Should come up with better approach to interpolating fare
        df.loc[df.fare==0.0, 'fare'] = .01
        
        globals()[filename] = df
        
    return globals()[filename]


def get_split(dataname = 'x_train'):
    
    if dataname not in globals():
        
        train_raw = get_raw('train')
        
        x_train, x_test, y_train, y_test = train_test_split(train_raw,
                                                            train_raw.survived,
                                                            random_state = 1)
        
        globals()['x_train'] = x_train
        globals()['x_test'] = x_test
        globals()['y_train'] = y_train
        globals()['y_test'] = y_test
    
    return globals()[dataname]

#   get_split()


#----------------------------------------------------------
#   Build tidy dataset
#   This must be suitable for both training and test data
#       as well as the final eval data
#----------------------------------------------------------
def tidy_data(data):
    
    #   Fill in missing age values based on model
    data.loc[data.age.isnull(), 'age'] = model_age(data)

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
    data['is_alone'] = (data.fam_size==0).astype(int)
    data['has_cabin'] = (data.cabin.isnull()).astype(int)
    
    #   Create categorical variables
    for col in ['pclass', 'sibsp', 'parch', 'embarked', 'fam_size', 'ticket_class', 'cabin_floor']:
        data[col] = data[col].astype('category')

    return data


def get_tidy(p_type = 'train'):
    
    filename = p_type + '_tidy'
    
    if filename not in globals():
        
        if p_type == 'eval':
            df = get_raw('eval')
        else:
            df = get_split('x_' + p_type)
        
        print('Loading ' + filename)

        tidy = tidy_data(df.copy())
        
        globals()[filename] = tidy
        
    return globals()[filename]

#   get_tidy('train')
#   get_tidy('test')
#   get_tidy('eval')
#   get_tidy().dtypes


def build_model_age():
    
    x_train = get_split('x_train')
    
    if 'predict_age' not in globals():
            
        formula = 'age ~ sex + pclass'
    
        model_age = sm.ols(formula, data = x_train).fit()
        
        globals()['predict_age'] = model_age
        
    return globals()['predict_age']

#   build_model_age()

def model_age(df):
    
    model = build_model_age()

    age_pred = model.predict(df[df.age.isnull()])
    
    return np.ceil(age_pred)
    
#   model_age(get_split('x_train'))


    