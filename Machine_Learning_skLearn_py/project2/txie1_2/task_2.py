##Tianchu Xie 
# 113148828
# Project 2

from sklearn import tree
import	matplotlib.pyplot	as	plt	
import	numpy	as	np	
from	sklearn	import	datasets	
from	sklearn.cross_validation	import	train_test_split	
from	sklearn.naive_bayes	import	GaussianNB	
from	sklearn.metrics	import	accuracy_score,	confusion_matrix	
import csv
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cross_validation import train_test_split
from numpy import genfromtxt
from sklearn import datasets
import warnings


            
           
            



def main():
    
    try:
        my_data = genfromtxt('train.csv', delimiter = ',')
        data_l = np.delete(my_data, 57, axis = 1)
        data_r = my_data[:, 57]
    except:
        warnings.warn("deprecated", DeprecationWarning)
        
    X_train, X_val, y_train, y_val = train_test_split(data_l, data_r, test_size=0.26)	
    new_data = tree.DecisionTreeClassifier(min_samples_split = 28, min_samples_leaf = 7)
    new_data.fit(X_train, y_train)
    
    acc = new_data.predict(X_val)
    print(accuracy_score(y_val, acc)) 
    #the acc = 0.913528591353, actually, I got most 0.93*, but it's been covered by new data sets  
    
    test_data = genfromtxt('ts2.csv', delimiter = ',')
    ret = [[]]
  
    ret = new_data.predict(test_data).T
    prob = new_data.predict_proba(test_data)
    
        
    np.savetxt('pred_2.csv', ret, delimiter = ',')
    np.savetxt('prob_2.csv', prob, delimiter = ',')
    
    
    
  #  cmap = {0: 'cyan', 1:'yellow'}
  #  colors = [cmap[x] for x in ret]
   # plt.figure(figsize = (20, 15))
   # for i in range (8):
   #     for j in range(8):
   #         plt.subplot(8, 8, i * 8 + j + 1)
   #         plt.scatter(test_data[:, i], test_data[:, j], c = colors)
    #        plt.xlabel(i)
    #        plt.ylabel(j)
   # plt.show()
    #this is how I print the result
    


if __name__ == '__main__':
    main()
    

