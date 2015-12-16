#! /usr/bin/python
import sys
def reducer():
    '''
    Given the output of the mapper for this assignment, the reducer should
    print one row per weather type, along with the average value of
    ENTRIESn_hourly for that weather type, separated by a tab. You can assume
    that the input to the reducer will be sorted by weather type, such that all
    entries corresponding to a given weather type will be grouped together.

    In order to compute the average value of ENTRIESn_hourly, you'll need to
    keep track of both the total riders per weather type and the number of
    hours with that weather type. That's why we've initialized the variable 
    riders and num_hours below. Feel free to use a different data structure in 
    your solution, though.

    An example output row might look like this:
    'fog-norain\t1105.32467557'
    '''

    riders = 0      # The number of total riders for this key
    num_hours = 0   # The number of hours with this key
    old_key = None

    for line in sys.stdin:
        # split District and values
        line_split = line.strip().split("\t")
        this_key,entries_values = line_split
        # If you have a new key the print out the old key and the sum of all its values then set new key and values
        if old_key and old_key != this_key:
            print"{0}\t{1}".format(old_key,riders/float(num_hours))
            old_key = this_key
            riders = float(entries_values)
            num_hours = 1
        # If this is the first input the establish the first key  and value 
        elif not old_key:
            old_key = this_key
            riders = float(entries_values)
            num_hours = 1
        # for identical keys add values to running total for the key(District)
        else: 
            riders += float(entries_values)
            num_hours += 1
    
    print"{0}\t{1}".format(old_key,riders/float(num_hours)) 


if __name__=="__main__":
    reducer()