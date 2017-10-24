##Tianchu Xie 
# 113148828
# Project 1


import	matplotlib.pyplot	as	plt	
import	numpy	as	np	
from	sklearn	import	datasets	
from	sklearn.cross_validation	import	train_test_split	
from	sklearn.naive_bayes	import	GaussianNB	
from	sklearn.metrics	import	accuracy_score,	confusion_matrix	
from sklearn import svm
from sklearn import preprocessing
from scipy import interp
import csv
from numpy import genfromtxt
import warnings
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold




def main():
    
    try:
        my_data = genfromtxt('train.csv', delimiter = ',')
        data_l = genfromtxt('sel1.csv', delimiter = ',')
        #data_l = np.delete(my_data, 57, axis = 1)
        data_r = my_data[:, 57]
    except:
        warnings.warn("deprecated", DeprecationWarning)
        
        
    test_data = genfromtxt('ts3.csv', delimiter = ',')
    sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
    #sel = sel.fit_transform(data_l)

   # np.savetxt('sel1.csv', sel, delimiter = ',')
    
    
    X_train, X_val, y_train, y_val = train_test_split(data_l, data_r, test_size=0.33, random_state = 9)	
    
    X_trainScale = preprocessing.scale(X_train)
    new_data = svm.SVC(kernel='linear', C=2, random_state = 9, probability = True)
    new_data.fit(X_trainScale, y_train)
    X_valScale = preprocessing.scale(X_val)
    
    acc = new_data.predict(X_valScale)
    print(accuracy_score(y_val, acc)) #the acc = 0.931868131868
    print(confusion_matrix	(y_val,acc))
    
    
    ret = [[]]
  
    ret = new_data.predict(test_data).T
    prob = new_data.predict_proba(test_data)
    
        
    np.savetxt('pred_3.csv', ret, delimiter = ',')
    np.savetxt('prob_3.csv', prob, delimiter = ',')
    
    
    #mean_tpr = 0.0
    #mean_fpr = np.linspace(0, 1, 100)

    #colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    #lw = 2

    # Compute ROC curve and ROC area for each class

    
  # fpr, tpr, thresholds = roc_curve(, prob[:,1])

if __name__ == '__main__':
    main()
    

