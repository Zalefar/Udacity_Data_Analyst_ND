#! /usr/bin/python
import sys
def reducer():
    '''
    Given the output of the mapper for this exercise, the reducer should PRINT 
    (not return) one line per UNIT along with the total number of ENTRIESn_hourly 
    over the course of May (which is the duration of our data), separated by a tab.
    An example output row from the reducer might look like this: 'R001\t500625.0'

    You can assume that the input to the reducer is sorted such that all rows
    corresponding to a particular UNIT are grouped together.
    '''
    # Establish key and value
    Tot_Entries_by_hr = 0
    old_Key = None
    
    for line in sys.stdin:
        # split District and values
        line_split = line.strip().split("\t")
        this_Key,entries_values = line_split
        # If you have a new key the print out the old key and the sum of all its values
        if old_Key and old_Key != this_Key:
            print"{0}\t{1}".format(old_Key,Tot_Entries_by_hr)
            old_Key = this_Key
            Tot_Entries_by_hr = float(entries_values)
        # If this is the first input the establish the first key  and value 
        elif not old_Key:
            old_Key = this_Key
            Tot_Entries_by_hr = float(entries_values)
        # for identical keys add values to running total for the key(District)
        else: 
            Tot_Entries_by_hr += float(entries_values)
    
    print"{0}\t{1}".format(old_Key,Tot_Entries_by_hr)

if __name__=="__main__":
    reducer()