#! /usr/bin/python
import sys
def mapper():
    """
    In this exercise, for each turnstile unit, you will determine the date and time 
    (in the span of this data set) at which the most people entered through the unit.
    
    The input to the mapper will be the final Subway-MTA dataset, the same as
    in the previous exercise. You can check out the csv and its structure below:
    https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv

    For each line, the mapper should return the UNIT, ENTRIESn_hourly, DATEn, and 
    TIMEn columns, separated by tabs. For example:
    'R001\t100000.0\t2011-05-01\t01:00:00'
    """

    for line in sys.stdin:
        # split input lines by ','
        line_split = line.strip().split(",")
        # ignore header line from csv file
        if line_split[1] == 'UNIT':
            continue
        else:
            print"{0}\t{1}\t{2}\t{3}".format(line_split[1],line_split[6],line_split[2],line_split[3])
            
if __name__=="__main__":
    mapper()