import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as sm


FOLDERS = {'raw': 'data_raw/',
           'clean': 'data_clean/'}


def get_feature_set(name):
    
    features = pd.read_csv('data_raw/features.csv')
    
    return features.loc[features[name]==1, 'feature'].values


#--------------------------------------
#   Modeling Functions for Tidy Data
#--------------------------------------
def build_model_age():
    
    data = get_split('x_train').copy()
    
    if 'predict_age' not in globals():

        data['title_master'] = data.name.str.contains('Master') * 1
        
        for col in ['pclass', 'parch']:
            encode = pd.get_dummies(data[col], prefix = col)
            data = pd.concat([data, encode], axis = 1)
    
        formula = 'age ~ title_master + pclass_3 + parch_2 + parch_0 + pclass_1'
    
        model_age = sm.ols(formula, data = data[data.age.notnull()]).fit()
        
        globals()['predict_age'] = model_age
        
    return globals()['predict_age']

def model_age(df):
    
    data = df.copy()

    model = build_model_age()

    age_pred = model.predict(data)
    
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
    
    data['name_length'] = data.name.str.len()
    data['title'] = data.name.str.extract(' ([A-Za-z]+)\.')
    mr = ['Rev', 'Dr', 'Col', 'Capt', 'Don', 'Major']
    mrs = ['Mme', 'Countess', 'Lady']
    miss = ['Mlle']

    data.loc[data.title.isin(mr), 'title'] = 'mr'
    data.loc[data.title.isin(mrs), 'title'] = 'mrs'
    data.loc[data.title.isin(miss), 'title'] = 'miss'
    data.loc[data.title=='Master', 'title'] = 'master'
    data.loc[~data.title.isin(['mr', 'mrs', 'miss', 'master']), 'title'] = ''
    
    data['cabin_floor'] = data.cabin.str.replace('[0-9]| ', '').str.get(0).str.lower()
    data.loc[data.cabin_floor.isnull(), 'cabin_floor'] = 'z'
    
    data['ticket_alpha'] = data.ticket.str.extract('([A-Za-z\.\/]+)').str.replace('\.', '').str.lower()
    data['ticket_num'] = data.ticket.str.extract('([0-9\.\/]+)').str.replace('\.', '')

    data['fam_size'] = data.sibsp + data.parch + 1
    data['is_alone'] = ((data.sibsp + data.parch) == 0) * 1

    data.loc[data.embarked.isnull(), 'embarked'] = 'c'
    data.embarked = data.embarked.str.lower()

    for col in ['pclass', 'sex', 'sibsp', 'parch', 'embarked', 'title', 'cabin_floor', 'fam_size']:
        encode = pd.get_dummies(data[col], prefix = col)
        data = pd.concat([data, encode], axis = 1)

    #   Fill in missing fare values based on linear model
    data['fare_gen'] = data.fare
    data.loc[(data.fare == 0) | (data.fare.isnull()), 'fare_gen'] = model_fare(data[(data.fare == 0) | (data.fare.isnull())])

    #   Fill in missing age values based on linear model
    data['age_gen'] = data.age
    data.loc[data.age_gen.isnull(), 'age_gen'] = model_age(data[data.age_gen.isnull()])

    #   Place ages into bins    
    bins = [0, 14, 32, 99]

    data['age_bin'] = pd.cut(data['age_gen'], bins, labels = ['age_0_14', 'age_14_32', 'age_32_99'])
    enc_age = pd.get_dummies(data.age_bin)
    data = pd.concat([data, enc_age], axis = 1)

    #   Scale age and fare using StandardScaler
    std_scale = StandardScaler().fit(data[['age_gen', 'fare_gen', 'name_length']])
    data['age_scaled'] = 0
    data['fare_scaled'] = 0
    data['name_scaled'] = 0
    data[['age_scaled', 'fare_scaled', 'name_scaled']] = std_scale.transform(data[['age_gen', 'fare_gen', 'name_length']])

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

