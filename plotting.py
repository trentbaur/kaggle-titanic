import matplotlib.pyplot as plt

def hist(x, p_bins = 10):
    
    plt.hist(x, bins = p_bins)


#   plt.hist(x_train[x_train.fare.notnull()].fare, bins = 20)
#   plt.hist(x_train[(x_train.fare.notnull()) & (x_train.fare < 150) & (x_train.fare > 20)].fare, bins = 30)
