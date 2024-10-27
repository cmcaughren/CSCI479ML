#Carolyn McAughren A4 CSCI479 Machine Learning October 30 2022
import sys
import pandas as pd
import numpy as np
import math

class LinearRegressionModel:
   def __init__(self, learning_rate, w, max_iteration, error_threshold):
      self.learning_rate = learning_rate
      self.w = w
      self.max_iteration = max_iteration
      self.error_threshold = error_threshold
      self.model_error = None

   def gradient_descent_algorithm(self, data):
      print("Gradient Descent Algorithm Output")
      print(f"Starting weight values:")
      print(f"w[0]: {self.w[0]}, w[1]: {self.w[1]}, w[2]: {self.w[2]}\n")
      iteration = 1
      while (iteration <= self.max_iteration):
         self.error(data) #calculates and sets self.model_error
         if self.model_error < self.error_threshold:
            break
         for i in range(0, len(self.w)):
            adjust = self.error_delta(data, i)
            self.w[i] = (self.w[i] + (self.learning_rate*adjust) )
         print(f"Iteration: {iteration}")
         print(f"Model Error: {self.model_error}")
         print(f"w[0]: {self.w[0]}, w[1]: {self.w[1]}, w[2]: {self.w[2]}\n")
         iteration += 1
      print("Gradient Descent Algorithm Final Output:")
      data = self.predict_oxycon(data)
      print(data.to_string())
   
   #calculate overall error for the model
   def error(self, data):
      err = 0
      for i in range(1, (len(data) + 1)): #iterates through all items in dataset
         model_output = (self.w[0] + (self.w[1])*(data.loc[i]['AGE']) + (self.w[2])*(data.loc[i]['HEARTRATE']))
         expected_output = data.loc[i]['OXYCON']
         err = err + ((expected_output - model_output)**2) #power of 2
      self.model_error = (0.5)*err

   #calculate the adjustment amount for weight w[i]
   def error_delta(self, data, i):
      delta = 0
      for j in range(1, (len(data) + 1)):  #iterates through all items in dataset
         model_output = (self.w[0] + (self.w[1])*(data.loc[j]['AGE']) + (self.w[2])*(data.loc[j]['HEARTRATE']))
         expected_output = data.loc[j]['OXYCON']
         
         if (i == 0):
            xij = 1 #w[0] has no attribute partner, so x is 1
         else: 
            xij = data.loc[j].iloc[i - 1] #attribute value which partners with weight w[i], for item j in dataset
         delta = delta + ( (expected_output - model_output)*(xij))
      return delta

   #once model is built, predict oxycon for all data items and add into data as new column
   def predict_oxycon(self, data):
      data = data.assign(PREDICTION_OXYCON = lambda x: (self.w[0] + self.w[1]*x['AGE'] + self.w[2]*x['HEARTRATE']))
      data = data.assign(DIFF = lambda x: (x['PREDICTION_OXYCON'] - x['OXYCON']))
      return data


def main():
   if len(sys.argv) is not 2:
      print(f"ERROR: Expecting 1 command line arguements: \n \
         A csv file location containing the data, \n \
         ex. 'python3 A4.py A4-data.csv ")
      quit()
   data = read_in_data()
   data = process_data(data)
   w = [-59.5, -0.15, 0.60]
   model = LinearRegressionModel(0.000002, w, 10, 3.0)
   model.gradient_descent_algorithm(data)
   
def read_in_data():
   try:
      data = pd.read_csv(sys.argv[1])
      return data
   except FileNotFoundError:
      print(f"\nERROR: The file {sys.argv[n]} does not exist.")
      quit()

def process_data(data):
   data = pd.DataFrame(data)
   data = data.set_index('ID')
   return data

if __name__ == "__main__":
   main()
