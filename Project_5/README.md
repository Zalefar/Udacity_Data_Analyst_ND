#Project 5 Identify Fraud from Enron Email   
###Author: Zach Farmer
The files contained in this directory contain all of the resources and functions engaged in the pursuit of this  project. A list of files can be found below along with a brief description of their content and purpose.    

Project_5_Machine_Learning_POI_Classifier.ipynb:
* IPython notebook containing the project in its entirety. All of the request components are contained within. This was done to show the flow of thought in the onstruction and completion of this project.      

final_project_dataset.pkl:         
* The dataset provided by Udacity that was used to train my classifier.        

Best_Classifiers.pkl:         
* A pickle python object containing my final classifier (in this case entire pipeline) used to find potential candidates who should be examined closer for fraudulent behavior. This object can be generated in the provided poi_id.py script but does take some time to run so I have provided it here.         

create_my_dataset.py:        
* Intermediate Functions used in the creation of my classifer. I have written outside the ipython notebook because they are used in my parallelizable which is run on an aws ec2 cluster. Therefore the intermediate code needed to be portable.      
 
ec2_cluster_files/:       
* A directiory designed as container for all the files to be ssh'd to my aws ec2 cluster file system for the purpose of implementing my parallelizable gridsearch and classifer trainer.        

emails_by_address/:        
* Directory of email addresses for all the emails associated with the Enron Fraud case. This directory was provided by Udacity and is used in my noetbook for data description purposes only.      
 
enron61702insiderpay.pdf:       
* pdf version of the financial information used mainly for reference.     

my_classifier.pkl:      
* Trained classifer (pipeline object) as a pickled object.   
    
my_dataset.pkl:        
* The final version of the dataset used in the creation of the classifier.      

my_feature_list.pkl:        
* The final list of features used in the creation of the classifier.       

poi_id.py:        
* The final script used to train and return the final dataset, feature list and trained classifier.       

poi_names.txt:       
* Udacity provided list of hand coded persons of interest.      

select_features.py:        
* Intermediate functions used in the selection of my features. Functions focus on return the metrics used to determine the value of my features towards the training of my classifier.   
    
tester.py:        
* Udacity provided script for evaluating the performance of the trained classifier.   
    
tools/:       
* Udacity provided directory containing functions that are used in the processing of the dataset into numpy arrays of features and labels.       
  

