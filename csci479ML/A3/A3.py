#Carolyn McAughren A3 CSCI479 Machine Learning October 21 2022

import sys
import pandas as pd
import numpy as np
import math


def main():
   if len(sys.argv) is not 2:
      print(f"ERROR: Expecting 1 command line arguements: \n \
         A csv file location containing the data, \n \
         ex. 'python3 lab3.py CarData.csv ")
      quit()
   data = ReadInData()
   data = ProcessData(data)

   #build probability tables for each attribute
   buyProb = CalcProbability('BUY', data)
   maintProb = CalcProbability('MAINT', data)
   doorsProb = CalcProbability('DOORS', data)
   personsProb = CalcProbability('PERSONS', data)
   lugProb = CalcProbability('LUG_BOOT', data)
   safetyProb = CalcProbability('SAFETY', data)

   #create the eval table
   len_data = len(data)
   rowValues = data['EVALUATION'].unique()
   evalProb = pd.DataFrame(np.zeros((len(rowValues), 1)))
   evalProb.index = rowValues
   evalProb.columns = ["Probability"]
   for attr in rowValues:
      evalProb.loc[attr]["Probability"] = (len(data[data['EVALUATION'] == attr])) / len_data
   
   #write tables to file
   with open("probTables.txt", 'w') as f:
      print("PROBABILITY TABLES", file=f)
      print("\nEVALUATION", file=f)
      print(F"{evalProb}", file=f)
      print("\nBUY", file=f)
      print(f"{buyProb.to_string()}", file=f)
      print("\nMAINTENENCE", file=f)
      print(f"{maintProb.to_string()}", file=f)
      print("\nDOORS", file=f)
      print(f"{doorsProb.to_string()}", file=f)
      print("\nPERSONS", file=f)
      print(f"{personsProb.to_string()}", file=f)
      print("\nLUG BOOT", file=f)
      print(f"{lugProb.to_string()}", file=f)
      print("\nSAFETY", file=f)
      print(F"{safetyProb.to_string()}", file=f)
   
   evalPred = PredictEval(data, buyProb, maintProb, doorsProb, personsProb, lugProb, safetyProb, evalProb)
   with open("predTable.txt", 'w') as f:
      print("PREDICTION TABLE\n", file=f)
      print(F"{evalPred.to_string()}", file=f)

   accuracyTable = CalculateAccuracy(data, evalPred)
   with open("accTable.txt", 'w') as f:
      print("ACCURACY TABLE\n", file=f)
      print(F"{accuracyTable.to_string()}", file=f)

def ReadInData():
   try:
      data = pd.read_csv(sys.argv[1])
      return data
   except FileNotFoundError:
      print(f"\nERROR: The file {sys.argv[n]} does not exist.")
      quit()

def ProcessData(data):
   data = pd.DataFrame(data)
   data = data.set_index('ID')
   return data
   
def CalcProbability(attribute, data):

   #get a list of unique values this attribute can take on
   rowValues = data[attribute].unique()
   colValues = data['EVALUATION'].unique()

   #create new dataframe filled with 0's of correct size to fit all probabilities for this attrib.
   #set those unique attribute values as row headers, and n counts and p counts as column headers
   probTable = pd.DataFrame(np.zeros((len(rowValues), len(colValues))))
   probTable.index = rowValues
   probTable.columns = colValues
   
   #calculate the percentage of each
   #using Laplace smoothing
   k = 2
   for row in rowValues:
      for col in colValues:
         #number of items with Ai = Vi AND EVAL = C
         numRowCol = len( data[(data[attribute] == row) & (data['EVALUATION'] == col)])
         #number of items with EVAL = C
         numCol = len(data[data['EVALUATION'] == col]) 
         #number of Vi values which Attribute Ai could take on 
         numRow =  len(rowValues)
         probTable.loc[row][col] = ((numRowCol + k) / (numCol + (numRow*k)))

   return probTable

def PredictEval(data, buyProb, maintProb, doorsProb, personsProb, lugProb, safetyProb, evalProb):
   
   evalPred = pd.DataFrame(np.zeros((len(data), 1)))
   evalPred.index = data.index
   evalPred.columns = ["PREDICTED EVALUATION"]

   for i, row in data.iterrows():

      #calculate a probability for each level of Evaluation
      probUnacc = CalcPredProb("unacc", i, data, buyProb, maintProb, doorsProb, personsProb, lugProb, safetyProb, evalProb)
      probAcc = CalcPredProb("acc", i, data, buyProb, maintProb, doorsProb, personsProb, lugProb, safetyProb, evalProb)
      probGood = CalcPredProb("good", i, data, buyProb, maintProb, doorsProb, personsProb, lugProb, safetyProb, evalProb)
      probVgood = CalcPredProb("vgood", i, data, buyProb, maintProb, doorsProb, personsProb, lugProb, safetyProb, evalProb)
      
      #prediction for Evaluation is the max calculated probability for the levels of Evaluation
      maxProb = max(probUnacc, probAcc, probGood, probVgood)
      if maxProb == probUnacc:
         evalPred.loc[i] = "unacc"
      elif maxProb == probAcc:
         evalPred.loc[i] = "acc"
      elif maxProb == probGood:
         evalPred.loc[i] = "good"
      else:
         evalPred.loc[i] = "vgood"
   return evalPred

#using Naive Bayes Classifier, calculate the probability of a level for Evaluation
def CalcPredProb(value, i, data, buyProb, maintProb, doorsProb, personsProb, lugProb, safetyProb, evalProb):
   return ((buyProb.loc[data.loc[i]["BUY"]][value]) * (maintProb.loc[data.loc[i]["MAINT"]][value]) *\
            (doorsProb.loc[data.loc[i]["DOORS"]][value]) * (personsProb.loc[data.loc[i]["PERSONS"]][value]) *\
            (lugProb.loc[data.loc[i]["LUG_BOOT"]][value]) * (safetyProb.loc[data.loc[i]["SAFETY"]][value]) *\
            (evalProb.loc[value]["Probability"]))

def CalculateAccuracy(data, evalPred):

   data['predEval'] = evalPred

   rowHeader = ["actual_unacc", "actual_acc", "actual_good", "actual_vgood"]
   colHeader = ["pred_unacc", "pred_acc", "pred_good", "pred_vgood"]
   accuracyTable = pd.DataFrame(np.zeros((4,4)))
   accuracyTable.index = rowHeader
   accuracyTable.columns = colHeader

   evalValues = ["unacc", "acc", "good", "vgood"]
   for row in evalValues:
      for col in evalValues:
         accuracyTable.loc["actual_" + row]["pred_" + col] = len(data[(data["EVALUATION"] == row) & (data["predEval"] == col)])
   return accuracyTable


      


if __name__ == "__main__":
   main()
