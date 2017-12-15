import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def hist(x, p_bins = 10):
    
    plt.hist(x, bins = p_bins)


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
