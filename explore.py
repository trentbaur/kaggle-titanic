import pandas as pd
import gather_data, plotting

def display_nulls():
    
    train_raw = gather_data.load_data()
    
    #   Explore null values in dataset
    for x in train_raw.columns.values:
        nulls = train_raw[x].isnull()
        
        if sum(nulls) > 0:
            print('Null counts for ' + x)
            print(nulls.value_counts()[True])
            print()
    return

#   display_nulls()


def display_counts():
    
    train_raw = gather_data.load_data()

    #   Explore value counts in dataset
    for x in train_raw.columns.values:
        if len(train_raw[x].unique()) < 10:
            print('Counts for ' + x)
            print(train_raw[x].value_counts())
            print()
    
    return

#   display_counts()


def display_survival_counts():
    
    train_raw = gather_data.load_data()

    #   Explore survival counts for easily grouped columns
    for x in train_raw.columns.values:
        if (x != 'Survived') & (len(train_raw[x].unique()) < 10):
            print('Survival Counts for ' + x)
            print(pd.crosstab(train_raw['Survived'], train_raw[x]))
            print()
    
    return

#   display_survival_counts()


def explore_data():
    
    display_nulls()
    
    print('--------------------\n')
    
    display_counts()
    
    print('--------------------\n')
    
    display_survival_counts()
    
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
