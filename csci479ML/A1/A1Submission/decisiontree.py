"""
   Carolyn McAughren A1 CSCI 479 Machine Learning

 * Pre-process the data and transform the "Measurement" attribute to a categorical one 
   using one of the algorithms/programs developped in Lab 1 and Lab 2.

 * Design and implement an information based learning algorithm to build a decision tree using 
   the given data set.

   The given data set "A1-data.csv" contains 5,200 instances and has descriptive attributes:
   "Supply_Level", "Valve_Position", and "Measurement". Based on these attributes, a
   decision tree is built to preduct "STATUS".
"""
import sys
import pandas as pd
import numpy as np
import math
from lab2 import best_split

def main():
   if len(sys.argv) is not 2:
      print(f"ERROR: Expecting 1 command line arguements: a filename containing the data to be split. \n \
         ex. 'python3 decisiontree.py A1-data.csv' ")
      quit()

   data = read_in_data()
   attribute = confirm_attribute(data)
   print_state_bool = False #to prevent extra output messages from lab2.py
   updated_data = best_split(data, attribute, print_state_bool)

   #drop unneeded columns
   del updated_data['Event_ID']
   del updated_data['Measurement']

   print("DECISION TREE\n")
   build_binary_tree(updated_data, 0, "root", "entire dataset", {})
   

#confirms the data file exists
def read_in_data():
   try:
      data = pd.read_csv(sys.argv[1])
      return data
   except FileNotFoundError:
      print(f"\nERROR: The file {sys.argv[1]} does not exist.")
      quit()

#check that "Measurement" attribute is in the data
def confirm_attribute(data):
   attribute = "Measurement"
   columns = data.columns.values
   if attribute not in columns:
      print(f"'Measurement' is not a column header in the provided data.")
      quit()
   return attribute


#using ID3 style algorithm, create a decision tree by selecting the attribute with the 
#lowest entropy at each branch of the tree.
#Output messages are displayed as the tree is build so that it can be visualized.
def build_binary_tree(data, level, nodeNum, nodeNumTotal, knownAttributes):

   numItems = len(data)
   numNItems = len(data[data["STATUS"] == 'Normal'])
   numCItems = len(data[data["STATUS"] == 'Critic'])
   misclassification_rate = 1 - max((numNItems/numItems), (numCItems/numItems))
   
   #allows tree output messages to have nested indentation
   if level != 0: 
      spacing = "----"*level
   else:
      spacing = ""
   
   #output messages detailing what the distribution of items is at this node
   print(f"{spacing}Level: {level}, value '{nodeNum}' of attribute '{nodeNumTotal}'")
   print(f"{spacing}Number Items: {numItems}, Number of Normal Status Items: {numNItems}, Number of Critical Status Items: {numCItems}")
   print(f"{spacing}Misclassification Rate is: {round(misclassification_rate, 5)} ")
   print(f"{spacing}Known Attributes: {knownAttributes}")

   #Basecase checks
   if numNItems == 0:
      print(f"{spacing}STATUS KNOWN: CRITICAL\n")
      return
   elif numCItems == 0:
      print(f"{spacing}STATUS KNOWN: NORMAL\n")
      return
   elif len(knownAttributes) == 3:
      print(f"{spacing}No more valid attributes to continue splitting data!")
      best_guess = "Normal" if (numNItems >= numCItems) else "Critical"
      print(f"{spacing}Based on the distribution of status in items of this leafnode, we make a best-guess.")
      print(f"{spacing}STATUS MOST LIKELY: {best_guess}\n")
      return

   #determine the attribute with the lowest entropy as the next split 
   splitAttribute, entropy = find_best_split(data)
   attributeValues = data[splitAttribute].unique() #find all values this attribute can take on
   print(f"{spacing}NEXT BEST SPLIT SELECTED: {splitAttribute} with an Entropy of {round(entropy, 5)}")

   print("\n")

   for value in attributeValues:

      #add newly assigned value for this attribute for displaying
      newKnownAttributes = knownAttributes.copy()
      newKnownAttributes[splitAttribute] = value

      #remove this attribute as a candidate for further splits
      updatedData = data[data[splitAttribute] == value]
      del updatedData[splitAttribute]
      
      #recursively continue to build the tree
      build_binary_tree(updatedData, level + 1, value, splitAttribute, newKnownAttributes)

#try all candidate splits and pick one with lowest entropy
def find_best_split(data):

   total_items = len(data)
   possible_splits = data.columns.copy()
   possible_splits = possible_splits.drop("STATUS") #cannot split on target value 

   #calculate entropy for each potential split attribute,remember the name of lowest entropy,
   #and the value of that entropy
   best_split = None
   best_split_entropy = math.inf #pos infinity
   
   #we wish to calculate the entropy for each value this attribute can take on, then sum them together
   #to find entropy for the entire split
   for split in possible_splits:   #for each candidate attribute to split the remaining data
      
      entropy = 0

      values = data[split].unique()
      for value in values: #for each value which the data can take on for this attribute
         
         currData = data[data[split] == value] #slice of data with this value for attribute, for easier calculations
         valueCount = len(currData) #number of items with attribute = value
         valueNCount = len(currData[currData["STATUS"] == "Normal"]) #number of items with attribute = value and status = normal
         valueCCount = len(currData[currData["STATUS"] == "Critic"]) #number of items with attribute = value and status = critical
         
         # calculate entropy for this segment of the split, add to running sum
         if (valueCount == 0) or (valueNCount == 0) or (valueCCount == 0):
            valueEntropy = 0
         else: 
            valueEntropy = (valueCount/total_items)*( (-1)*(valueNCount/valueCount)*(math.log(valueNCount/valueCount, 2)) + (-1)*(valueCCount/valueCount)*(math.log(valueCCount/valueCount, 2)) )
         entropy = entropy + valueEntropy

      if entropy < best_split_entropy:
         best_split = split
         best_split_entropy = entropy

   return best_split, best_split_entropy
   

if __name__ == "__main__":
   main()