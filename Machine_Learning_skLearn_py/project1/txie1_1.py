##Tianchu Xie 
# 113148828
# Project 1


import	matplotlib.pyplot	as	plt	
import	numpy	as	np	
from	sklearn	import	datasets	
from	sklearn.cross_validation	import	train_test_split	
from	sklearn.naive_bayes	import	GaussianNB	
from	sklearn.metrics	import	accuracy_score,	confusion_matrix	
import csv
from numpy import genfromtxt
import warnings



def main():
    
    try:
        my_data = genfromtxt('train.csv', delimiter = ',')
        data_l = np.delete(my_data, 57, axis = 1)
        data_r = my_data[:, 57]
    except:
        warnings.warn("deprecated", DeprecationWarning)
        
    X_train, X_val, y_train, y_val = train_test_split(data_l, data_r, test_size=0.33)	
    pgaussian = GaussianNB()
    pgaussian.fit(X_train, y_train)
    
    acc = pgaussian.predict(X_val)
    print(accuracy_score(y_val, acc)) #the acc = 0.830769230769
    print(confusion_matrix	(y_val,acc))
    
    test_data = genfromtxt('ts1.csv', delimiter = ',')
    ret = [[]]
  
    ret = pgaussian.predict(test_data).T
        
    np.savetxt('pred_2.csv', ret, delimiter = ',')

if __name__ == '__main__':
    main()
    

