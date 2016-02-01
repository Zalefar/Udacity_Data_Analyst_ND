#!/usr/bin/python
#Author: Zach Farmer
#Purpose: Generate pkl files containing my dataset, list of features, and final
#optimized classifier

import sys
import numpy as np
import os
import pickle
import re
import scipy.stats as sp

from pprint import pprint
from create_my_dataset import newFeatures, dropFeatures
sys.path.append("tools/")
from feature_format import featureFormat, targetFeatureSplit  

from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA 

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.externals import joblib
from IPython.parallel import Client
from sklearn.grid_search import ParameterGrid
from sklearn.grid_search import RandomizedSearchCV

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV

from sklearn.metrics import classification_report

def dropFeatures(features, remove_list):
    """
        Parameters: 
            features = Python list of unique features in the data_dict. 
            remove_list = Python list of features to be removed.(drop 
            non-numeric features such as the email address) 
    
        Output: 
            Python list of unique features sans the features in the remove
            list. 
    """   
    ## Method courtesy of user: Donut at: 
    ## http://stackoverflow.com/questions/4211209/remove-all-the-elements-that-occur-in-one-list-from-another
    features_remove = remove_list
    learning_features = [feature for feature in features if feature not in features_remove]
    
    return learning_features

## Following code adapted from Udacity's Intro to Machine learning lesson 11 Visualizing
## Your New Feature Quiz
def computeFraction(feature_1, feature_2):
    """ 
        Parameters: 
            Two numeric feature vectors for which we want to compute a ratio
            between
     
        Output: 
            Return fraction or ratio of feature_1 divided by feature_2
    """
    
    fraction = 0.
    
    if feature_1 == "NaN":  
        fraction = 0.0
    elif feature_2 == "NaN":
        fraction = 0.0
    else: 
        fraction = int(feature_1) / float(feature_2)

    return fraction
     
def newFeatures(data_dict):
    """
        Parameters: 
            data_dict provided by Udacity instructors   
    
        Output: 
            data_dict with new features (hard-coded)
    """
    ## following is not extensible or abstractable to making any other features 
    ## then what is hard coded below.
    for name in data_dict:

        from_poi_to_this_person = data_dict[name]["from_poi_to_this_person"]
        to_messages = data_dict[name]["to_messages"]
        fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
        
        data_dict[name]["fraction_from_poi"] = fraction_from_poi


        from_this_person_to_poi = data_dict[name]["from_this_person_to_poi"]
        from_messages = data_dict[name]["from_messages"]
        fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    
        data_dict[name]["fraction_to_poi"] = fraction_to_poi
        
        salary = data_dict[name]['salary']
        total_payments = data_dict[name]['total_payments'] 
        salary_to_totalPayment_ratio = computeFraction(salary, total_payments)
        
        data_dict[name]['salary_to_totalPayment_ratio'] = salary_to_totalPayment_ratio
        
        salary = data_dict[name]['salary']
        total_stock_value = data_dict[name]['total_stock_value'] 
        salary_to_stockValue_ratio = computeFraction(salary, total_stock_value)
        
        data_dict[name]['salary_to_stockValue_ratio'] = salary_to_stockValue_ratio
        
    return data_dict   


def PickleBestClassifers(best_classifiers, file_Name):
    """
        Parameters:
            best_classifiers = A python dictionary containing the names of 
            classifers as keys and a pipeline object containing the optimized 
            paramters for the feature selection and classifier. 
            file_name = The name that the pickled file will be saved under 
            as a python string.
            
        Output:
            (none) pickled object saved to the local directory.
    """     
    # Pickle the results
    fileObject = open(file_Name,'wb') 
    
    pickle.dump(best_classifiers, fileObject)   
    
    fileObject.close()
    print "{0} saved to local directory as a pickle file".format(file_Name)
    
    return None

def removeOutliers(data_dict,listOfOutliers):
    """
        Parameters:
            data_dict= The data_dict provided by Udacity. 
            listOfOutliers = Python List of outliers (key names)
        to remove from the data_dict.
        
        Output:
            Updated data_dict where the outliers have been removed.
    """
    for outlier in listOfOutliers:
        try:
            data_dict.pop(outlier,0)
        except ValueError:
            pass
    
    return data_dict

def generateFeaturesList(my_dataset):
    """
        Parameters:
            my_dataset = Updated data_dict which includes the new features and has had 
        outliers removed. 
        
        Output:
            A python list containing all of the features to be used in the fitting and 
        testing of the classifier.
    """
    ## Find unique features in my_dataset
    features = [value for value in my_dataset.itervalues() for value in value.keys()]
    unique_features = list(set(features))
    
    ## Remove non-numeric features (email_address)
    reduced_features = dropFeatures(unique_features, ['email_address'])
    
    ## Method for moving an item in a list to a new position found at:
    ## http://stackoverflow.com/questions/3173154/move-an-item-inside-a-list
    ## posted by nngeek
    ## ensure that 'poi' is the first value in the feature list
    try:
        reduced_features.remove('poi')
        reduced_features.insert(0, 'poi')
    except ValueError:
        pass
        
    return reduced_features

if __name__=="__main__":
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)

    ## following if statement to be run only if the optimized classifier/feature 
    ## Select pipeline object is not found in a the local directory in the pickle file. 
    ## This block of code will rerun the entire grid search and pipeline process to 
    ## generate the content that should be available in the pickle file.
    if "Best_Classifiers.pkl" not in os.listdir('.'):    
        ## set random seed generator for the sciy.stats    
        np.random.seed(42)

        ## Remove Outliers
        data_dict = removeOutliers(data_dict,['Total'])
        
        ## Create new features 
        my_dataset = newFeatures(data_dict) 
        
        ## Generate the feature list for featureFormat
        reduced_features = generateFeaturesList(my_dataset)
        
        ## Extract features and labels from dataset
        data = featureFormat(my_dataset, reduced_features, sort_keys=True)
        ## Return as numpy arrays    
        labels, numpy_features = targetFeatureSplit(data)

        ## Create training and test splits on all of the features, feature 
        ## selection to be performed in the pipeline 
        X_train, X_test, y_train, y_test = train_test_split(numpy_features,\
                                                            labels,\
                                                            test_size=0.15,\
                                                            random_state=42)
        
        ## set randomized grid search cv
        cv = StratifiedShuffleSplit(y_train,\
                                    n_iter = 50,\
                                    test_size = .3,\
                                    random_state=42)
    
        
        classifiers = {
                   "GNB": GaussianNB(), 
                   "SVC": svm.SVC(),
                   "RDF": RandomForestClassifier(),
                   "ADB": AdaBoostClassifier(DecisionTreeClassifier(class_weight='balanced')),
                   "LRC": LogisticRegressionCV(random_state = 42,)
                       }
       
        ## dictionary of parameters for the randomized grid search cv 
        param_grid = dict(
                    features__pca__n_components = sp.randint(1,len(X_train[0])),
                    features__univ_select__percentile = range(1,100,10),
                    SVC__C = sp.expon(scale = 100),
                    SVC__gamma = sp.expon(scale=.1),
                    SVC__kernel = ['rbf', 'linear','sigmoid'],
                    SVC__class_weight = ['balanced'],
                    RDF__n_estimators = range(1,500,1),
                    RDF__criterion = ['gini','entropy'],
                    RDF__max_depth = range(1,len(X_train[0]),1),
                    RDF__class_weight = ['balanced'],
                    ADB__n_estimators = range(1,500,1),
                    ADB__learning_rate = sp.expon(scale= 300),
                    LRC__Cs = range(0,10,1),
                    LRC__class_weight = ['balanced']
                            )    

        best_classifiers = {}

        for classifier in classifiers:
            ## Method for supplying just the parameter grid entries related to the classifier
            ## in the current interation while excluding the other classifer paramters.
            # dict comprehension method courtesy of BernBarn at:
            # http://stackoverflow.com/questions/14507591/python-dictionary-comprehension
            param_for_class = {key: value for key,value in param_grid.iteritems() if
                                 re.search(key.split("_")[0],'features ' + classifier)}
            
            ## Feature selection method, same for all classifiers
            pca = PCA()
            selection = SelectPercentile()
            
            ## Note: Only implement when using randomized grid search. PCA takes a long
            ## time to run, not a good choice with exhaustive grid search
            feature_select = FeatureUnion([("pca",pca),("univ_select",selection)])
            
            ## Active the classifier for the current loop
            clf = classifiers[classifier]
        
            ## Pipeline feature selection, feature scaling and classifier for optimization
            pipeline = Pipeline([
                                ("features", feature_select),
                                ("scaler", MinMaxScaler()),
                                (classifier,clf)
                                ])
            
            ## use f1_weighted scoring to account for heavily skewed classes
            search = RandomizedSearchCV(estimator = pipeline,
                                        param_distributions = param_for_class,
                                        scoring = 'f1_weighted',
                                        n_jobs=-1,
                                        cv = cv,
                                        n_iter = 20,
                                        verbose = 1,
                                        error_score = 0,
                                        random_state = 42)
            
            results = search.fit(X_train,y_train)
            best_classifiers[classifier] = results.best_estimator_

        ## Save the best classifier pipeline objects to local directory using pickle
        PickleBestClassifers(best_classifiers,"Best_Classifiers.pkl")
    else:
        ## After initial run of grid search, reference the pickled outcomes for the 
        ## rest of the analysis. Actual searching process takes some time
        ## on my system setup, so I want to run it as few times as possible. 
        savedResults = open("Best_Classifiers.pkl",'r')  
        best_classifiers = pickle.load(savedResults)  

## Remove Outliers
data_dict = removeOutliers(data_dict,['Total'])

### Store to my_dataset for easy export below.
my_dataset = newFeatures(data_dict) 

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = generateFeaturesList(my_dataset)

## Find best classifier
clf = best_classifiers["LRC"]

### Dump classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features_list)
 