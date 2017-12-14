import pandas as pd
import math 
import pdb

FOLDERS = {'raw': 'data_raw/',
           'clean': 'data_clean/'}


def load_data(type = 'train.csv'):

    return pd.read_csv(FOLDERS['raw'] + type)

#   load_data()


def tidy_data(data):

    data.columns = [col.lower() for col in data.columns]

    #pdb.set_trace()
    
    data.name = data.name.replace('Ms.', 'Miss', regex = True)
    
    for title in ('Master', 'Miss', 'Mr.', 'Mrs.', 'Dr'):
        med = math.ceil(data[(data.name.str.contains(title)) & (data.age.notnull())].age.describe()['50%'])
        
        data.loc[data.age.isnull() & data.name.str.contains(title), 'age'] = med
        

    enc_sex = pd.get_dummies(data.sex)
    data = data.join(enc_sex)
    data.drop(['sex'], axis = 1, inplace = True)

    data.embarked = data.embarked.str.lower()
    
    enc_embark = pd.get_dummies(data.embarked)
    data = data.join(enc_embark)
    data.drop(['embarked'], axis = 1, inplace = True)

    data.loc[data.fare.isnull(), 'fare'] = data.fare.describe()['50%']
        
    data['ticket_class'] = data.ticket.str.replace('[0-9]| ', '')

    data['cabin_floor'] = data.cabin.str.replace('[0-9]| ', '').str.get(0)

    return data



