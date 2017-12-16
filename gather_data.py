import pandas as pd
import numpy as np
import math 
import pdb


FOLDERS = {'raw': 'data_raw/',
           'clean': 'data_clean/'}

def load_data(data_type = 'train'):
    if data_type + '_raw' not in globals():
        print('Loading ' + data_type + '_raw')
        
        df = pd.read_csv(FOLDERS['raw'] + data_type + '.csv')
        
        df.columns = [col.lower() for col in df.columns]

        globals()[data_type + '_raw'] = df
        
    return globals()[data_type + '_raw']


def load_y():
    train_raw = load_data()
    
    if 'y_data' not in globals():
        
        print('Loading y_data')
        
        globals()['y_data'] = train_raw.survived
        
    return globals()['y_data']


def tidy_data(data_type = 'train'):

    if data_type + '_tidy' not in globals():
        
        print('Loading ' + data_type + '_tidy')

        data = load_data(data_type).copy()
        
        #   Survived will exist in training data but not dev data
        if 'survived' in data.columns:
            data.drop(labels = ['survived'], axis = 1, inplace = True)
    
        data.name = data.name.replace('Ms.', 'Miss', regex = True)
        
        for title in ('Master', 'Miss', 'Mr.', 'Mrs.', 'Dr'):
            med = math.ceil(data[(data.name.str.contains(title)) & (data.age.notnull())].age.describe()['50%'])
            
            data.loc[data.age.isnull() & data.name.str.contains(title), 'age'] = med
    
        bins = [0, 12, 18, 25, 40, 60, 75]
    
        #pdb.set_trace()
        
        data['age_bin'] = pd.cut(data['age'], bins, labels = ['age_0_12', 'age_12_18', 'age_18_25', 'age_25_40', 'age_40_60', 'age_60_75'])
        enc_age = pd.get_dummies(data.age_bin)
        data = pd.concat([data, enc_age], axis = 1)
    
        enc_sex = pd.get_dummies(data.sex)
        data = data.join(enc_sex)
        
        data['age*male'] = data.age * data.male
    
        data.embarked = data.embarked.str.lower()
        
        enc_embark = pd.get_dummies(data.embarked)
        data = data.join(enc_embark)
    
        data.loc[data.fare.isnull(), 'fare'] = data.fare.describe()['50%']
        
        data['fare_log'] = np.log(data.fare)
        
        data['fam_size'] = data.sibsp + data.parch
        
        data['ticket_class'] = data.ticket.str.replace('[0-9]| ', '')
    
        data['cabin_floor'] = data.cabin.str.replace('[0-9]| ', '').str.get(0).astype('category')
    
        enc_floor = pd.get_dummies(data.cabin_floor)
        data = data.join(enc_floor)

        data.pclass = data.pclass.astype('category')
        data.sibsp = data.sibsp.astype('category')
        data.parch = data.parch.astype('category')
        data.embarked = data.embarked.astype('category')
                
        globals()[data_type + '_tidy'] = data

    return globals()[data_type + '_tidy']

#   tidy_data()
#   tidy_data().dtypes

def load_combined_tidy():
    train_tidy = tidy_data('train')
    y_data = load_y()
    
    return pd.concat([train_tidy, y_data], axis = 1)
    
    

