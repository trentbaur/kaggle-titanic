

#--------------------------------------------------
#   Generate metrics to evaluate model performance
#--------------------------------------------------

"""
from sklearn.metrics import classification_report

print(classification_report (y_train,
                             y_train_pred,
                             target_names = ['Survived', 'Died']))

print(classification_report (y_test,
                             y_test_pred,
                             target_names = ['Survived', 'Died']))

plotting.show_roc(y_train, y_train_pred)
plotting.show_roc(y_test, y_test_pred)
"""





from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate


for x in range(1, 20):
    print(x * 50)
    
    model = GradientBoostingClassifier(n_estimators = x * 50,
                                       learning_rate = .05)
    
    scores = cross_val_score(model, x_train_thin, y_train, cv = 5)
    print(scores, scores.mean())



