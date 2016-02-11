# Project 2 NYC Subway Analysis   
The files contained in this directory are all related to the pursuit of analyzing New York Subway traffic data as it related to weatherunderground provided weather statistics. The are several IPython notebooks included one which contains the entire project from its origins to the final conclusions and a second which contains only the final conclusions and analysis.       

`DA_Project_1.{html...ipynb}`:
	Ipython and .html version of the project in its entirety. The notebook utilizes the other files included in this directory and if you have an installation of anaconda installed you should be able to run the ipy. notebook on your personal machine. Project includes all my steps and notes, meant to document thought process and methodological approach, not meant to be a presentation. 

`Short-Q&A_Project_1.{html...ipynb}`:
	Presentable version of the project containing summarys and answers to the Udacity provided questions.   

`Subway_avg_results.txt`:
	Created as a result of using linux bash to act as proxy for a mapreduce job run on an HDFS.    

`Subway_data_avg_mapper.py`:
	Python script containing a mappers for a Map reduce job. For the ultimate purpose of finding averages, data returned in this script designed to be passed on the a reducer .py script.   

`Subway_data_avg_reducer.py`:
	Complementary reducer to the the average data mapper .py script.

`Subway_data_max_mapper.py`: 
	Very similar to previous mapper only this ones is designed to find max values.

`Subway_data_max_reducer.py`: 
	Complementary reducer to max value mapper.

`Subway_data_mapper`:
	.py Mapreduce mapper function for summary purposes.

`Subway_data_reducer`:
	complementary reducer to the prior mapper.

`master_turnstile.csv`:
	Original NYC subway turnstile data without weather data. Used in exploratory data analysis.

`turnstile_data.txt`: 
	Updated version of master turnstile data with 'corrections' applied. 

`updated_turnstile_data.txt`:
	Another further updated version of the turnstile data.

`turnstile_data_master_with_weather.csv`:
	Data set containing a both the updated turnstile data and the weather data used for the final summaries and conclusions.  

`weatherUnderground.csv`: 
	Weather data from weatherunderground for the dates in the turnstile data.    


