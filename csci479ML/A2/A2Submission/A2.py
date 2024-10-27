#Carolyn McAughren A3 CSCI479 Machine Learning October 13 2022

import sys
import pandas as pd
import numpy as np
import math


def main():
   if len(sys.argv) is not 3:
      print(f"ERROR: Expecting 2 command line arguements: \n \
         A csv file location containing the training data, \n \
         and a second csv file location containing the testing data, \n \
         ex. 'python3 lab3.py A2-training-data.csv A2-testing-data.csv' ")
      quit()
   trainingData = ReadInData(1)
   testingData = ReadInData(2)
   
   #convert data so one user is one vector, drop columns not useful
   trainingData, testingData = ProcessData(trainingData, testingData)

   #compare each user in test data to each user in training data, measure distance
   #find recommendations based on the k nearest neighbors
   for i, rowTest in testingData.iterrows():
      distances = []
      for j, rowTrain in trainingData.iterrows():
         distances.append(DistanceCalculation(testingData.loc[i], trainingData.loc[j]))
      distances = pd.DataFrame(distances)
      distances.columns = ["distance"]
      KNN(i, distances, trainingData)
      

def ReadInData(n):
   try:
      data = pd.read_csv(sys.argv[n])
      return data
   except FileNotFoundError:
      print(f"\nERROR: The file {sys.argv[n]} does not exist.")
      quit()

def ProcessData(trainingData, testingData):
   trainingData = trainingData.drop(columns = ['Unnamed: 4', 'Unnamed: 5', 'date_of_grade'])
   testingData = testingData.drop(columns = ['Unnamed: 4', 'Unnamed: 5', 'date_of_grade'])

   #create blank dataframe to the size needed, correct index to match user id's for that range
   #add in normalized movie ratings under each movie column id
   newTrainingData = pd.DataFrame(np.zeros((trainingData['user'].nunique(), trainingData['movie'].nunique())))
   newTrainingData.index = trainingData['user'].unique()
   for index, row in trainingData.iterrows():
      grade = Normalize(row['grade'])
      newTrainingData[row['movie']][row['user']] = grade

   #process testing data, keeping same column headers (movie ids) as those found in training data
   newTestingData = pd.DataFrame(np.zeros((testingData['user'].nunique(), trainingData['movie'].nunique())))
   newTestingData.index = testingData['user'].unique()
   for index, row in testingData.iterrows():
      grade = Normalize(row['grade'])
      newTestingData[row['movie']][row['user']] = grade

   return newTrainingData, newTestingData
   

#simple normalization calculation, hard coded for this example
def Normalize(grade):
   return ( (grade - 1) / 4 )

#using Euclidean distance formula:
#EuclideanDistance = sqrt(sum for i to N (userA[i] – userB[i])^2)
def DistanceCalculation(userA, userB):
   sum = 0
   for i in range(len(userA)):
      sum = sum + (float(userA[i]) - float(userB[i]))**2
   euclideanDistance = math.sqrt(sum)
   return euclideanDistance
   

def KNN(i, distances, trainingData):

   #sort all distances between user i in testing data, and all users in training data
   distances = distances.sort_values('distance')
   #keep only the 10 nearest neighbors, smallest distance values kept
   KNNdistances = distances[:10]
   movieSuggestions = []
   movieSuggestionList = []
   
   #select any movie rated 5 stars by anyone in the k nearest neighbors ratings
   for index, row in KNNdistances.iterrows():
      movieSuggestions = trainingData.loc[[index]]      
      movieSuggestions = movieSuggestions.loc[:, [(movieSuggestions[col] == 1).any() for col in movieSuggestions.columns]].columns
      movieSuggestionList.extend(movieSuggestions)
      

   movieSuggestionList = pd.DataFrame(movieSuggestionList)
   movieSuggestionList.columns = ["Movie"]
   print(f"Suggestions for user {i}:")

   #obtain counts of movies rated 5 stars, see if any were rated highly by multiple users
   movieSuggestionList = movieSuggestionList.groupby('Movie').size().reset_index(name='Count')
   #sort output to display movies rated 5 stars multiple times first, then sort by Movie id
   movieSuggestionList = movieSuggestionList.sort_values(by=['Count','Movie'],ascending=[False,True]).reset_index(drop=True)
   
   print(movieSuggestionList)


if __name__ == "__main__":
   main()
