import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as sm


FOLDERS = {'raw': 'data_raw/',
           'clean': 'data_clean/'}

COLS_TO_MODEL = [#'pclass', 
                 'pclass_1', 'pclass_2', 'pclass_3', 
                 #'age',
                 #'age_gen',
                 #'age_norm',
                 'age_scaled',
                 #'age_0_16', 'age_16_25', 'age_25_40', 'age_40_60', 'age_60_75',
                 #'sibsp',
                 #'sibsp_0', 'sibsp_1', 'sibsp_2', 'sibsp_3', 'sibsp_4', 'sibsp_5', 'sibsp_8',
                 #'parch',
                 #'parch_0', 'parch_1', 'parch_2', 'parch_3', 'parch_4', 'parch_5', 'parch_6', 
                 #'fam_size',
                 'fam_size_1', 'fam_size_2', 'fam_size_3', 'fam_size_4', 'fam_size_5', 'fam_size_6', 'fam_size_7', 'fam_size_8', 'fam_size_11',
                 #'fare',
                 #'fare_gen',
                 #'fare_log',
                 #'fare_norm',
                 'fare_scaled',
                 #'is_alone',
                 #'has_cabin',
                 #'embarked',
                 #'name_len',
                 'name_scaled',
                 'embarked_c', 'embarked_q', 'embarked_s',
                 #'cabin_floor',
                 #'cabin_floor_A', 'cabin_floor_B', 'cabin_floor_C', 'cabin_floor_D', 'cabin_floor_E', 'cabin_floor_F', 'cabin_floor_G', 'cabin_floor_T',
                 #  Gender dummy variables, placed last to avoid comma-commenting issues
                 #'sex_female',
                 'sex_male']

    
#--------------------------------------
#   Modeling Functions for Tidy Data
#--------------------------------------
def build_model_age():
    
    x_train = get_split('x_train')
    
    if 'predict_age' not in globals():
            
        formula = 'age ~ pclass + sibsp + parch'
    
        model_age = sm.ols(formula, data = x_train).fit()
        
        globals()['predict_age'] = model_age
        
    return globals()['predict_age']

def model_age(df):
    
    model = build_model_age()

    age_pred = model.predict(df)
    
    return np.ceil(age_pred)


def build_model_fare():
        
    if 'predict_fare' not in globals():

        x_train = get_split('x_train')
        
        formula = 'fare ~ pclass + sibsp + parch'
    
        model_fare = sm.ols(formula, data = x_train).fit()
        
        globals()['predict_fare'] = model_fare
        
    return globals()['predict_fare']

def model_fare(df):
    
    model = build_model_fare()

    fare_pred = model.predict(df)
    
    return round(fare_pred, 2)
    

#-----------------------------
#   Data retrieval functions
#-----------------------------
def get_raw(data_type = 'train'):
    
    filename = data_type + '_raw'
    
    if filename not in globals():
        
        print('Loading ' + filename)
        
        df = pd.read_csv(FOLDERS['raw'] + data_type + '.csv')
        
        df.columns = [col.lower() for col in df.columns]
        
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


#----------------------------------------------------------
#   Build tidy dataset
#   This must be suitable for both training and test data
#       as well as the final eval data
#----------------------------------------------------------
def tidy_data(data):
    
    #   Fill in missing age values based on linear model
    data['age_gen'] = data.age
    data.loc[data.age_gen.isnull(), 'age_gen'] = model_age(data[data.age_gen.isnull()])

    #   Fill in missing fare values based on linear model
    data['fare_gen'] = data.fare
    data.loc[(data.fare == 0) | (data.fare.isnull()), 'fare_gen'] = model_fare(data[(data.fare == 0) | (data.fare.isnull())])

    #   Place ages into bins    
    bins = [0, 16, 25, 40, 60, 75]
   
    data['age_bin'] = pd.cut(data['age_gen'], bins, labels = ['age_0_16', 'age_16_25', 'age_25_40', 'age_40_60', 'age_60_75'])
    enc_age = pd.get_dummies(data.age_bin)
    data = pd.concat([data, enc_age], axis = 1)

    data.loc[data.embarked.isnull(), 'embarked'] = 'c'
    data.embarked = data.embarked.str.lower()

    data['cabin_floor'] = data.cabin.str.replace('[0-9]| ', '').str.get(0).astype('category')

    #   Create interaction variables
    data['age_norm'] = data.age_gen / data.age_gen.max()
    data['fare_norm'] = data.fare_gen / data.fare_gen.max()
    data['fare_log'] = np.log(data.fare_gen)
    data['fam_size'] = data.sibsp + data.parch + 1
    data['ticket_class'] = data.ticket.str.replace('[0-9]| ', '')
    data['is_alone'] = (data.fam_size==0).astype(int)
    data['has_cabin'] = (data.cabin.isnull()).astype(int)
    data['name_len'] = data.name.apply(lambda x: len(x))

    #   Scale age and fare using StandardScaler
    std_scale = StandardScaler().fit(data[['age_gen', 'fare_gen', 'name_len']])
    data['age_scaled'] = 0
    data['fare_scaled'] = 0
    data['name_scaled'] = 0
    data[['age_scaled', 'fare_scaled', 'name_scaled']] = std_scale.transform(data[['age_gen', 'fare_gen', 'name_len']])

    for col in ['pclass', 'sex', 'sibsp', 'parch', 'fam_size', 'embarked', 'cabin_floor']:
        encode = pd.get_dummies(data[col], prefix = col)
        data = pd.concat([data, encode], axis = 1)

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
        
        if p_type != 'train':
            add_missing_columns(tidy)
            
        globals()[filename] = tidy
        
    return globals()[filename]

#   get_tidy('train')
#   get_tidy('test')
#   get_tidy('eval')


def add_missing_columns(df):
    #   For the model's sake, ensure that all of the dummy variables get created
    #   in both the test and eval datasets. Initialize to zero
    train_cols = get_tidy('train').columns
    
    missing = [col for col in train_cols if col not in df.columns]
    
    for x in missing:
        print('Adding ' + x)
        df[x] = 0
    
    return

