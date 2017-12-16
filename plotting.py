import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import gather_data

def show_pairs_plot():

    train_raw = gather_data.load_data()

    cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    
    axes = pd.plotting.scatter_matrix(train_raw[cols], alpha=0.2)
    
    plt.tight_layout()

    return

#   show_pairs_plot()

def show_boxplot_log_fare():
    df = gather_data.load_combined_tidy()

    fig = plt.figure(1, figsize=(9, 6))

    ax = fig.add_subplot(111)

    bplot = plt.boxplot([np.log(df[df.Survived==0].fare), np.log(df[df.Survived==1].fare)], patch_artist = True)

    for patch, color in zip(bplot['boxes'], ['red', 'lightgreen']):
        patch.set_facecolor(color)
            
    ax.set_xticklabels(['Died', 'Survived'])
    ax.set_title('Survival Boxplots of log(fare)')       
    plt.show
    
    return

#   show_boxplot_log_fare()
    
    
def hist(x, p_bins = 10):
    
    plt.hist(x, bins = p_bins)

    return

#   plt.hist(X_train.age, bins = 20)
#   plt.hist(X_train[y_train==0].age, bins = 20)
#   plt.hist(X_train[y_train==1].age, bins = 20)
    
#   plt.hist(x_train[x_train.fare.notnull()].fare, bins = 20)
#   plt.hist(x_train[(x_train.fare.notnull()) & (x_train.fare < 150) & (x_train.fare > 20)].fare, bins = 30)


def show_roc(y_dev, y_pred):
    
    fpr, tpr, _ = roc_curve(y_dev, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return
