COLS_TO_MODEL = ['pclass', 
                 'age*class',
                 'age',
                 'sibsp',
                 'parch',
                 'fam_size',
                 'fare',
                 #  Embarked dummy variables
                 'c', 'q', 's',
                 #  Cabin Floor dummy variables
                 #'A', 'B', 'C', 'D', 'E', 'F', 'G', 'T',
                 #'age_0_12', 'age_12_18', 'age_18_25', 'age_25_40', 'age_40_60', 'age_60_75'
                 #  Gender dummy variables, placed last to avoid comma-commenting issues
                 'female',
                 'male']

