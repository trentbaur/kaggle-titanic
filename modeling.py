import statsmodels.formula.api as sm

import gather_data

COLS_TO_MODEL = ['pclass', 
                 'pclass_1', 'pclass_2', 'pclass_3', 
                 'age',
                 'sibsp',
                 'sibsp_0', 'sibsp_1', 'sibsp_2', 'sibsp_3', 'sibsp_4', 'sibsp_5', 'sibsp_8',
                 'parch',
                 'parch_0', 'parch_1', 'parch_2', 'parch_3', 'parch_4', 'parch_5', 'parch_6', 
                 'fam_size',
                 #'fare',
                 'is_alone',
                 #'has_cabin',
                 'embarked_c', 'embarked_q', 'embarked_s',
                 'cabin_floor_A', 'cabin_floor_B', 'cabin_floor_C', 'cabin_floor_D', 'cabin_floor_E', 'cabin_floor_F', 'cabin_floor_G', 'cabin_floor_T',
                 #'age_0_12', 'age_12_18', 'age_18_25', 'age_25_40', 'age_40_60', 'age_60_75'
                 #  Gender dummy variables, placed last to avoid comma-commenting issues
                 'sex_female',
                 'sex_male']


def logreg_sig(p_vars = 'age * male'):
    
    df = gather_data.load_combined_tidy()
    
    formula = 'survived ~ ' + p_vars
    
    results = sm.ols(formula, data = df).fit()
    
    return results.summary()

#   logreg_sig()
#   logreg_sig('pclass + fare')
    

def build_model_age():
    
    if 'predict_age' not in globals():
        
        x_train = gather_data.get_data('x_train')
    
        formula = 'age ~ sex + pclass'
    
        model_age = sm.ols(formula, data = x_train).fit()
        
        globals()['predict_age'] = model_age
        
    return globals()['predict_age']

#   build_model_age()

def model_age(df):
    
    model = build_model_age()

    age_pred = model.predict(df[df.age.isnull()])
    
    return age_pred
    

#   model_age(gather_data.load_x_train())


