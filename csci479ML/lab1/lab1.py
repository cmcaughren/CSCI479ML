#Carolyn McAughren Lab 1 CSCI479 Machine Learning September 16/2022

import sys
import pandas as pd
import numpy as np
from math import ceil

def main():
   if len(sys.argv) is not 4:
      print(f"ERROR: Expecting 3 command line arguements: \n \
         first, a filename containing the data to be split, \n \
         second, the attribute title on which to split the data \n \
         third, a number for how many bins to split the continuous data into \n \
         ex. 'python3 lab1.py A1-data.csv Measurement 2' ")
      quit()

   data = read_in_data()
   attribute = confirm_attribute(data)
   numsplits = confirm_splits(data)
   updated_data = equal_width(data, attribute, numsplits)
   updated_data.to_csv('updated_data.csv')
   


def read_in_data():
   try:
      data = pd.read_csv(sys.argv[1])
      return data
   except FileNotFoundError:
      print(f"\nERROR: The file {sys.argv[1]} does not exist.")
      quit()

#check that attribute in the data
#return the attribute title or quit with error message if not
def confirm_attribute(data):
   attribute = sys.argv[2]
   columns = data.columns.values
   if attribute not in columns:
      print(f"Attribute provided is not a column header in the provided data.")
      quit()
   return attribute
   #print(f"{data[attribute].dtype}")?
   #confirm that attribute is numerical, if categorical we cant do it, quit

#checks that the number splits requested is valid 
def confirm_splits(data):
   numsplits = sys.argv[3]
   count = len(data)
   try:
      numsplits = int(numsplits)
      if numsplits > count:
         print(f"Number of splits exceeds number of items in the dataframe.\n")
         quit()
      if numsplits == 0:
         print(f"Number of splits must be greater than 0.\n")
         quit()
      return numsplits
   except ValueError:
      print(f"Number of splits requested is not an int.")
      quit()


def equal_width(data, attribute, numsplits):

   #find the total number of items in the dataframe
   count =len(data)

   #add error checking incase "count" is 0 for somereason.

   #calculate how many items of the total will go into each category
   # ie. "every 'rotate' number of items, increment the value assigned for new category "
   #we use ceiling rather than floor, so we do not end up with an extra category holding only one 
   #entry e.g. if count was 1201, split by 4 = 300.25. If we divide 1201 into groups of 300,
   #there will be 4 categories with 300 items each, then a 5th with 1 item. Rounding up gives us
   #3 categories with 301, and a 4th with 298, which is preferable and closer to equal bins.
   #not perfect TODO Improve this, as if numsplits is very high, not the greatest solution
   rotate = ceil(count / numsplits)

   #create name for new attribute
   new_attribute = attribute + "_categorical"

   #add new attribute column to the data
   data[new_attribute] = data.index // rotate

   #return new list with this added attribute
   return data





if __name__ == "__main__":
   main()
