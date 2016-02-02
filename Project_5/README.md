#Project 5 Identify Fraud from Enron Email   
###Author: Zach Farmer
The files contained in this directory contain all of the resources and functions utilized in the course of this project. A list of files can be found below along with a brief description of their content and purpose.    

`Project_5_Machine_Learning_POI_Classifier.ipynb`:
* IPython notebook containing the project in its entirety. All of the requested components are contained within. This was done to document the flow of thought in the construction of the project.     

`final_project_dataset.pkl`:         
* The dataset provided by Udacity that was used to train my classifier.        

`Best_Classifiers.pkl`:         
* A pickled python object containing my final classifiers (in this case a pipeline object for each of classifiers experimented on) used to find potential candidates who should be examined closer for fraudulent behavior. This object can be generated in the provided poi\_id.py script but does take some time to run so I have provided it here. The best classifier from this dictionary is the classifier in the final my\_classifier.pkl file        

`my_classifier.pkl`:      
* Final Trained classifer (pipeline object) as a pickled object.   
 
`my_dataset.pkl`:        
* The final version of the dataset used in the creation of the classifier.      

`my_feature_list.pkl`:        
* The final list of features used in the creation of the classifier.       

`poi_id.py`:        
* The final script used to train and return the final dataset, feature list and trained classifier. 

`create_my_dataset.py`:        
* Intermediate functions used in the creation of my classifer. Written here outside the ipython notebook because it's used in my parallelizable code which is run on an AWS EC2 cluster instance. Therefore this code needed to be easily portable.      

`select_features.py`:        
* Intermediate functions used in the selection of my features. Function focuses on returning the metrics used to determine the value of my features selected in the training of my classifier.    

`ec2_cluster_files/`:       
* A directiory designed as container for all the files to be ssh'd to an AWS EC2 cluster file system for the purpose of implementing my parallelizable grid-search for hyper-parameter optimization.        

`emails_by_address.tar.gz`:        
* Zipped directory of text files containing the email messages addressed to and from the email addresses in the file names. This directory was provided by Udacity and is used in my noetbook for data description purpose and was the directory used to generate the email related to and from message features in the data set. The directory is zipped because it contains over 1000 files.   
 
`tools/`:       
* Udacity provided directory containing functions that are used in the processing of the dataset into numpy arrays of features and labels.     

`enron61702insiderpay.pdf`:       
* pdf version of the financial information used mainly for reference to double check the outlier information and to review what the different financial features represented.     


`poi_names.txt`:       
* Udacity provided list of hand coded persons of interest.      
 
`tester.py`:        
* Udacity provided script for evaluating the performance of the trained classifier.   
 
`References.txt`: 
* Text file containing the references referred to during the course of the project.    

`Questions.md`:
* Markdown file containing all of the answers to the questions provided by the Udacity Instructors in one place. These answer are the same as those found throughout the IPyhton notebook containing the project. The questions and answers are provided here to satisfy the project requirements and to provide easier access rather then requiring reviewers to work through the whole ipython notebook.     
