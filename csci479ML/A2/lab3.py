#Carolyn McAughren Lab 3 CSCI479 Machine Learning September 16/2022

import sys
import pandas as pd
import numpy as np
import math

#todo grade normalization is hardcoded to "5" max rating
#create global constant to improve this

def main():
   if len(sys.argv) is not 2:
      print(f"ERROR: Expecting 1 command line arguements: \n \
         A csv file location containing the data to be split, \n \
         ex. 'python3 lab3.py A2-training-data.csv' ")
      quit()

   data = ReadInData()
   data = data.drop(columns = ['Unnamed: 4', 'Unnamed: 5', 'date_of_grade'])
   newData = ProcessData(data)

   #calculate distance between user 1 and 99
   #userA = 1
   
   #for i in range(2,100):
      #userB = i
      #distance = DistanceCalculation(newData[newData.index == userA], newData[newData.index == userB])
      #print(f"The distance between user 1 and {i} is {distance}")

def ReadInData():
   try:
      data = pd.read_csv(sys.argv[1])
      return data
   except FileNotFoundError:
      print(f"\nERROR: The file {sys.argv[1]} does not exist.")
      quit()

#data will be converted to a dictionary
#each key is a user
#each value is a List of Tuples: Each tuple is a movie, date, rating collection. 
def ProcessData(data):

   #using groupby, apply and lambda functions as they are more efficient than using forloop over a dataframe
   #userRatingsDict = data.groupby('user')[['movie', 'date_of_grade', 'grade']].apply(lambda g: list(tuple(g.values))).to_dict()

   numberUsers = data['user'].nunique()  # 1 to 990 = 100 users
   numberMovies = data['movie'].nunique() # 0 to 100 = 101 movies rated
   newData = pd.DataFrame(np.zeros((numberUsers + 1, numberMovies)))
   newData = newData.drop(0)
   
   for index, row in data.iterrows():
      grade = normalize(row['grade'])
      newData[row['movie']][row['user']] = grade

   #print(newData.to_string())
   return newData
   

def normalize(grade):
   return ( (grade - 1) / 4 )

#using Euclidean distance formula:
#EuclideanDistance = sqrt(sum for i to N (userA[i] – userB[i])^2)
def DistanceCalculation(userA, userB):
   sum = 0

   for i in range(len(userA.axes[1]) - 1):
      sum = sum + (float(userA[i]) - float(userB[i]))**2

   euclideanDistance = math.sqrt(sum)
   return euclideanDistance
   


if __name__ == "__main__":
   main()
