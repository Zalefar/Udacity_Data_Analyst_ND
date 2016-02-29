# Project 2 -- NYC Subway Analysis         
****
*Loaded, wrangled, analyzed and predicted NYC subway turnstile traffic. Utilizes statistical measures of similarity and data visualization to explore the data. Implements scikit-learn supervised learning algorithms to draw conclusions about weathers effect on subway traffic.*   

The files contained in this directory are related to the pursuit of analyzing New York Subway traffic data in connection with weatherunderground provided weather statistics. The are several IPython notebooks included, one which contains the entire project from its origins to the final conclusions and a second which contains only the final conclusions and analysis with answers to the Udacity provided questions. The notebook containing the project from beginning to end contains all of the Problem set found in the course related to the NYC Subway data.             

`DA_Project_2.{ipynb,html}`:         
	Ipython notebook and .html version of the project in its entirety. The notebook utilizes or creates the other files included in this directory and if you have an installation of anaconda installed you should be able to run the Ipython notebook on your personal machine. Project includes all my steps and notes, the notebook was created in the course of completing problem sets throughout Udacity's Intro to Data Science class. As a result the notebook's function is to provide the foundation upon which the Short Q&A notebook was developed from. You can find the code behind the numbers and anlysis in the Short Q&A here.  

`Short-Q&A_Project_2.{ipynb,html}`:               
	Presentable version of the project containing summaries and answers to the Udacity provided questions. CSS styling not available when viewing inside github file view. The analysis for these questions and answer was performed in the DA_Project_1 notebook.

`Subway_data_avg_mapper.py`:       
	Python script containing a mapper for a MapReduce job. For the ultimate purpose of finding averages, data returned in this script designed to be passed on the a reducer .py script.         

`Subway_data_avg_reducer.py`:      
	Complementary reducer to the the average data mapper .py script.     

`Subway_avg_results.txt`:           
	Created as the end result of using linux bash shell to act as proxy for a MapReduce job run on an HDFS (Hadoop Distributed File System). This text files contains the outcome of the data avg. mapper and reducer .py scripts.  

`Subway_data_max_mapper.py`:       
	Very similar to previous mapper only this ones is designed to find max values. Outcome returned to be passed to the corresponding .py reducer script.      

`Subway_data_max_reducer.py`:     
	Complementary reducer to max value mapper.     
	
`Subway_max_results.txt`:          
	Created as the end result of running the Subway data max mapper and reducer scripts.      

`Subway_data_mapper.py`:      
	.py script of a Mapreduce mapper function for summary purposes. Outcomes returned to be passed to the corresponding reducer script.       

`Subway_data_reducer.py`:         
	complementary reducer to the prior mapper.     
	
`Subway_data_results.txt`:      
	Created as the end result of runnning the Subway data mapper and reducer scripts.       

`master_turnstile.csv`:     
	Original NYC subway turnstile data without weather data. Used in exploratory data analysis.      

`turnstile_data.txt`:       
	Updated version of master turnstile data with 'corrections' applied.     
    
`updated_turnstile_data.txt`:       
	Further updated version of the turnstile data.     

`turnstile_data_master_with_weather.csv`:       
	Data set containing a both the updated turnstile data and the weather data used for the final summaries and conclusions.      

`weatherUnderground.csv`:        
	Weather data from weatherunderground for the dates in the turnstile data.         


