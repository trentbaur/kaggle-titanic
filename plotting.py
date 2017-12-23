import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc

from gather_data import get_raw, get_tidy


sns.set(style="whitegrid", color_codes=True)


#-----------------------------------
#   Visualizations for raw data
#-----------------------------------
def show_pairs_plot():

    train_raw = get_raw('train')

    cols = ['pclass', 'age', 'sibsp', 'parch', 'fare']
    
    pd.plotting.scatter_matrix(train_raw[cols], alpha=0.2)
    
    plt.tight_layout()

    return
#   show_pairs_plot()


def show_boxplot_log_fare():
    
    df = get_raw('train')

    fig = plt.figure(1, figsize=(9, 6))

    ax = fig.add_subplot(111)

    bplot = plt.boxplot([np.log(df[df.survived==0].fare), np.log(df[df.survived==1].fare)], patch_artist = True)

    for patch, color in zip(bplot['boxes'], ['red', 'lightgreen']):
        patch.set_facecolor(color)
            
    ax.set_xticklabels(['Died', 'Survived'])
    ax.set_title('Survival Boxplots of log(fare)')       
    plt.show
    
    return
#   show_boxplot_log_fare()


def show_hist(p_group = 'sibsp', p_x = 'age'):

    df = get_tidy('train')
    
    grouped = df.groupby([p_group, p_x, 'survived'], as_index = False).aggregate(len)
    grouped = grouped[[p_group, p_x, 'survived', 'passengerid']]
    grouped.columns = [p_group, p_x, 'survived', 'count']

    grid = sns.FacetGrid(grouped,
                         col = p_group,
                         col_wrap = 4,
                         hue = 'survived',
                         palette = ['red', 'green'])
    
    grid.set(ylim = (0, 30))
    
    grid.map(plt.bar, p_x, 'count', alpha = .7)
    
    return
#   show_hist(p_group = 'pclass', p_x = 'age')

    
#---------------------------------------
#   Visualizations for transformed data
#---------------------------------------
def show_barplot_survival(p_var = 'cabin_floor'):
    
    df = get_tidy('train')
    
    pd.crosstab(df['survived'], df[p_var]).T.plot(kind='bar')
    
    return
#   show_barplot_survival('embarked')
#   show_barplot_survival('pclass')
#   show_barplot_survival('age')


def show_dot_plot(p_x = 'cabin_floor', p_y = 'fare'):
    
    df = get_tidy('train')
    
    sns.stripplot(x = p_x,
                  y = p_y,
                  data = df,
                  jitter = True,
                  hue = 'survived',
                  palette = ['red', 'green'])
    
    return
#   show_dot_plot()
#   show_dot_plot('pclass', 'fare_gen')


#---------------------------------------
#   Visualizations for modeling results
#---------------------------------------
def show_barplot_cv(results, param = 'subsample'):
    
    fig = plt.figure()
    
    g = plt.barh(np.arange(results[param].shape[0]),
             results[param]['mean'],
             xerr = [results[param]['mean'] - results[param]['min'],
                     results[param]['max'] - results[param]['mean']],
             align='center',
             color='blue',
             ecolor='black')

    g[results[param]['mean'].idxmax()].set_color('green')
    
    plt.axis([results[param]['min'].min() * .98,
              results[param]['max'].max() * 1.02,
              -1,
              results[param].shape[0]])
    
    plt.yticks(np.arange(results[param].shape[0]), results[param].value)
    plt.xlabel(param)
    plt.title('Cross Validation')
    
    return

#show_barplot_cv(cv_results, 'n_estimators')
#show_barplot_cv(cv_results, 'subsample')
#show_barplot_cv(cv_results, 'learning_rate')


def show_barplot_all(results):
    
    for k in results.keys():
        show_barplot_cv(results, k)
    
    return

#show_barplot_all(cv_results)


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
