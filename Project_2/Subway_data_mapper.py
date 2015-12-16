#! /usr/bin/python
import sys

def mapper():
    """
    The input to this mapper will be the final Subway-MTA dataset, the same as
    in the previous exercise.  You can check out the csv and its structure below:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv

    For each line of input, the mapper output should PRINT (not return) the UNIT as 
    the key, the number of ENTRIESn_hourly as the value, and separate the key and 
    the value by a tab. For example: 'R002\t105105.0'
    """
    for line in sys.stdin:
        #split csv into values for each column
        line_split = line.strip().split(",")
        # Ignore header row
        if line_split[1] == 'UNIT':
            continue
        # print District(tab)Entries hourly
        else:
            print "{0}\t{1}".format(line_split[1],line_split[6])
               
if __name__=="__main__":
    mapper()