import pandas as pd

from gather_data import get_raw, get_split, get_tidy
import plotting

def display_nulls():
    
    train_raw = get_raw('train')
    
    #   Explore null values in dataset
    for x in train_raw.columns.values:
        nulls = train_raw[x].isnull()
        
        if sum(nulls) > 0:
            print('Null counts for ' + x)
            print(nulls.value_counts()[True])
            print()
    return


def display_counts():
    
    train_raw = get_raw('train')

    #   Explore value counts in dataset
    for x in train_raw.columns.values:
        if len(train_raw[x].unique()) < 10:
            print('Counts for ' + x)
            print(train_raw[x].value_counts())
            print()
    
    return


def display_survival_counts():
    
    train_raw = get_raw('train')

    #   Explore survival counts for easily grouped columns
    for x in train_raw.columns.values:
        if (x != 'survived') & (len(train_raw[x].unique()) < 10):
            print('Survival Counts for ' + x)
            print(pd.crosstab(train_raw['survived'], train_raw[x]))
            print()
    
    return


def display_age_by(p_group = 'pclass'):
    
    train_raw = get_raw()
    
    print(train_raw['age'].groupby([train_raw[p_group], train_raw['survived']]).mean())

    return

#display_age_by('parch')


def explore_data():
    
    display_nulls()
    
    print('--------------------\n')
    
    display_counts()
    
    print('--------------------\n')
    
    display_survival_counts()
    
    print('--------------------\n')
    
    display_age_by()
    
    plotting.show_pairs_plot()
    
    return

#   explore_data()


'''
name_stub = 'Master'

train[train.Name.str.contains(name_stub)]

import matplotlib.pyplot as plt
plt.hist(train[(train.Name.str.contains('Mrs')) & (train.Age.notnull())].Age, bins=10)

plt.hist(train[(train.age > 0)].fare, bins=10)

'''
