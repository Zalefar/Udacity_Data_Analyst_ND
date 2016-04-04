#!/usr/bin/python
#Author: Zach Farmer
#Purpose: Parallelize algorithm and feature selection with paramter grid search.
#Use numpy memarrays and Ipython Parallel run on AWS EC2 using StarCluster.   

"""
    The following code blocks borrower heavily from Olivier Grisel
    "Advanced Machine Learning with scikit-learn" tutorial given at PyCon
    2013 in Santa Clara. The tutorial provided ipython notebooks with
    examples, I have used the code examples in those notebooks as a guidline
    for implementing my gridsearch optimization in parallel. For those 
    interested you can find the notebooks and link to the tutorial at the 
    following github: https://github.com/ogrisel/parallel_ml_tutorial/
    
    some of the functions are adaptations of the code found in that tutorial.
    The code block was designed to be uploaded to a starcluster intialized ec2
    instance. (I hade trouble getting this code to work quickly on the ec2 instance,
    not sure why as the environment should be identical to my machine, just 
    with more cores. Regardless the distribution of computations didn't seem to 
    speed up the process, and I had to ditch the pca which basically caused the 
    computations to stall indefinitely) 
"""
import sys
import numpy as np
import os
import time
import pickle
import re

from pprint import pprint
from create_my_dataset import newFeatures, dropFeatures, removeOutliers, fixFinancialData
from feature_format import featureFormat, targetFeatureSplit  

from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import MinMaxScaler

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from IPython.parallel import Client
from sklearn.grid_search import ParameterGrid

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegressionCV


## Create Cross-validated training and test dataset using joblib
## borrowers heavily for Oliver grisels example
def persist_cv_splits(X, y, n_cv_splits = 14, filename="data",
                      suffix = "_cv_%03d.pkl", test_size=.1,
                      random_state=42):
    """
        Input: 
            X = The features data in a numpy array. 
            y = the corresponding ground truth labels.
            n_cv_splits = Number of cross_validated splits to make (folds). 
            filename = The filename prefix for the pickled splits. 
            suffix = The filename suffix for the pickled splits.
            (apply logical naming for easier interative access later 
            when performing model fitting. (e.g. data_cv_001,
            data_cv_002, ...)). 
            test_size = Number of data points to set aside for testing
            as a ratio. (.1,.2,...). 
            random_state = Number for the random state generator for
            replication.
            *Note: oweing to the small size of the data set and the rarity
            of a positive poi labels the cv split will be performed using a
            stratified shuffle split.
            
        Output:
            pickled cross-val datasets for use by numpy memory maps to reduce 
            redundant distributions of the dataset to each of the engines. 
            In the case of AWS Clusters the cross-val dataset should be shared
            across all the clusters using NFS.   
    """
    ## Implement stratified shuffle split 
    cv = StratifiedShuffleSplit(y, 
                                n_iter = n_cv_splits,
                                test_size = test_size,
                                random_state = random_state)
    ## List of cv_split filenames
    cv_split_filenames = []
    
    for i, (train, test) in enumerate(cv):
        # Create a tuple containing the cross_val fold 
        cv_fold = (X[train], y[train], X[test], y[test])
        # use the index for the filenaming scheme
        cv_split_filename = filename + suffix % i 
        # add absolute path to filename 
        cv_split_filename = os.path.abspath(cv_split_filename)
        # Using joblib dump the cv_dataset files 
        joblib.dump(cv_fold, cv_split_filename) 
        # add the name to the cv_split_filenames to pass on as an iterator
        cv_split_filenames.append(cv_split_filename)
        
    return cv_split_filenames

def compute_evaluation(cv_split_filename, model, params):
    """
        Parameters:
            cv_split_filename = The file name of the memory mapped numpy array 
            containing a fold of the cross-validated split on which the model is
            to be trained. 
            model = tuple containing in the [0] index the alias name for the classifier
            and in the [1] index the instantiation of the classifier itself.
            Params = A dictionary of relevant parameters for the pipeline objects.
        
        Output:
            The validation score for the pipeline for the given
            cross-val dataset and parameters.
    """
    from sklearn.externals import joblib  
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.feature_selection import SelectPercentile, f_classif
    from sklearn import svm 
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegressionCV

    X_train, y_train, X_validation, y_validation = joblib.load(
        cv_split_filename, mmap_mode='c')  
    
    ## Feature selection method, same for all classifiers
    selection = SelectPercentile()        
    
    ## Pipeline the feature selection, feature scaling and classifier for optimization
    ## procedure
    pipeline = Pipeline([
                        ("features", selection),
                        ("scaler", MinMaxScaler()),
                        (model[0],model[1])
                        ])
    
    # set model parameters
    pipeline.set_params(**params)  
    # train the model
    trained_pipeline = pipeline.fit(X_train, y_train)  
    # evaluate model score
    validation_score = trained_pipeline.score(X_validation, y_validation)
    
    return validation_score

def grid_search(lb_view, model,
                cv_split_filenames, param_grid):
    """
        Parameters: 
            lb_view = A load-balanced IPython.parallel client.
            model = tuple containing in the [0] index the alias name for the classifier
            and in the [1] index the instantiation of the classifier itself.
            cv_split_filenames = list of cross-val dataset filenames. 
            param_grid = dictionary of all the hyper-parameters for the pipeline
            objects to be trained.
       
        Output:
            List of parameters and list of asynchronous client tasks handles 
    """
    all_tasks = []   
    all_parameters = list(ParameterGrid(param_grid))
    
    for i, params in enumerate(all_parameters):
        task_for_params = []  
        
        for j,cv_split_filename in enumerate(cv_split_filenames):
            t = lb_view.apply(compute_evaluation, cv_split_filename,
                              model, params)
            task_for_params.append(t)  
            
        all_tasks.append(task_for_params)
        
    return all_parameters, all_tasks

def find_best(all_parameters, all_tasks, n_top=5):
    """compute the mean score of the completed tasks"""
    mean_scores = []
    
    for param, task_group in zip(all_parameters,all_tasks):
        scores = [t.get() for t in task_group if t.ready()]
        if len(scores) == 0:
            continue
        mean_scores.append((np.mean(scores), param))
        
    return sorted(mean_scores, reverse=True, key = lambda x: x[0])[:n_top]  

def progress(tasks):
    """
        Input: 
            The asynchronus task handles returned
        
        Output:
            The the number tasks that have been completed
    """
    return np.mean([task.ready() for task_group in tasks
                                 for task in task_group])

def PickleBestClassifers(best_classifiers, file_Name):
    """
        Input:
            A python dictionary containing the names of classifers as keys and the 
            pipeline object containing the optimized paramters for the feature selection
            and learning algorithms. The name that the pickled file will be saved
            under as a python string.
            
        Output:
            (None) pickled object saved to the local directory.
    """     
    # Pickle the results, as the gridSearch takes forever
    fileObject = open(file_Name,'wb') 
    
    pickle.dump(best_classifiers, fileObject)   
    
    fileObject.close()
    
    return None

if __name__ =="__main__":
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    
    ## set random seed generator for the sciy.stats    
    np.random.seed(42)

    ## Add new feature to my dataset
    my_dataset = newFeatures(data_dict) 
    
    ## Remove outliers
    my_dataset = removeOutliers(my_dataset,['TOTAL','THE TRAVEL AGENCY IN THE PARK'])
    
    ## Fix bad financial data
    my_dataset = fixFinancialData(my_dataset)
               
    ## Find unique features in my_dataset
    features = [value for value in my_dataset.itervalues() for value in value.keys()]
    unique_features = list(set(features))
                                
    ## Remove non-numeric features, return feature list (email_address)
    features_list = dropFeatures(unique_features, ['email_address'])
                                                      
    ## Method for moving an item in a list to a new position found at:
    ## http://stackoverflow.com/questions/3173154/move-an-item-inside-a-list
    ## posted by nngeek
    ## ensure that 'poi' is the first value in the feature list
    try:
      features_list.remove('poi')
      features_list.insert(0, 'poi')
    except ValueError:
      pass

    ### Extract features and labels convert to numpy arrays
    data = featureFormat(my_dataset, features_list, sort_keys=True)
    labels, numpy_features = targetFeatureSplit(data)
    
    ## Create training and test splits on all of the features, feature 
    ## selection to be performed in the pipeline 
    X_train, X_test, y_train, y_test = train_test_split(numpy_features,\
                                                        labels,\
                                                        test_size=0.2,\
                                                        random_state=42)
    
    
    ## Create training and test splits for the grid-search cross-validation
    cv_split_filenames = persist_cv_splits(np.array(X_train),\
                                           np.array(y_train),\
                                           n_cv_splits = 10,\
                                           filename="data",\
                                           suffix = "_cv_%03d.pkl",\
                                           test_size=.2,\
                                           random_state=42)

    if "Best_Classifiers.pkl" not in os.listdir('.'):    
        ## List of classifiers to explore and compare
        classifiers = {
                        "GNB": GaussianNB(), 
                        "SVC": svm.SVC(),
                        "RDF": RandomForestClassifier(),
                        "ADB": AdaBoostClassifier(DecisionTreeClassifier(class_weight='balanced')),
                        "LRC" : LogisticRegressionCV()  
                        }
       
        ## dictionary of parameters for the GridSearchCV 
        param_grid = dict(
                        features__percentile = range(10,100,10),
                        SVC__C = np.logspace(-2,8,9,base=3),
                        SVC__gamma = np.logspace(-9,3,9,base = 3),
                        SVC__kernel = ['rbf', 'linear','sigmoid'],
                        SVC__class_weight = ['balanced'],
                        RDF__n_estimators = range(10,100,10),
                        RDF__criterion = ['gini','entropy'],
                        RDF__max_depth = range(1,len(X_train[0]),1),
                        RDF__class_weight = ['balanced'],
                        ADB__n_estimators = range(50,500,50),
                        ADB__learning_rate = np.logspace(-2,8,9,base=3),
                        LRC__Cs = range(0,10,1),
                        LRC__class_weight = ['balanced']
                        )    

        best_classifiers = {}

        client  = Client(packer = "pickle")
        lb_view = client.load_balanced_view()
        
        for classifier in classifiers:
            ## Method for supplying just the parameter grid entries related to the classifier
            ## in the current interation while excluding the other classifer paramters.
            # dict comprehension method courtesy of BernBarn at:
            # http://stackoverflow.com/questions/14507591/python-dictionary-comprehension
            param_for_class = {key: value for key,value in param_grid.iteritems() if
                                 re.search(key.split("_")[0],'features ' + classifier)}
    
            lb_view.abort()
            time.sleep(4)
            
            model = (classifier, classifiers[classifier])
    
            all_parameters, all_tasks = grid_search(lb_view, model,\
                                                    cv_split_filenames,\
                                                    param_for_class)
            
            while progress(all_tasks) < .99:
                print("Tasks completed for {0}: {1}%".format(classifier,100 * progress(all_tasks)))
                time.sleep(30)
            
            
            [t.wait() for tasks in all_tasks for t in tasks]
        
            best_classifiers[classifier] = find_best(all_parameters,\
                                                     all_tasks,\
                                                     n_top = 1)
            
        PickleBestClassifers(best_classifiers,"Best_Classifiers.pkl")
    else:
        ## After initial run of grid search, reference the pickled outcomes for the 
        ## rest of the analysis. Actual searching process takes a while
        ## on my system setup, so I want to run it as few times as possible. 
        savedResults = open("Best_Classifiers.pkl",'r')  
        best_classifiers = pickle.load(savedResults)
        