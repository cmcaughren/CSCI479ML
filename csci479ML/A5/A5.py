#Carolyn McAughren A4 CSCI479 Machine Learning Nov 27 2022
import sys
import pandas as pd
import numpy as np
import math
import random


def main():
   if len(sys.argv) is not 2:
      print(f"ERROR: Expecting 1 command line arguements: \n \
         A csv file location containing the data, \n \
         ex. 'python3 A5.py Points.csv ")
      quit()
   data = read_in_data()
   data = process_data(data)

   K = 2
   N = 100
   data, C1_centroid, C2_centroid, n = K_means(data, K, N)
   
   #write data to file
   with open("result.txt", 'w') as f:
      print("POINTS DATA", file=f)
      print(F"{data.to_string()}", file=f)
      print(f"Centroids stabilized in iteration {n}.", file=f)
      print(f"Final C1 centroid = {C1_centroid}", file=f)
      print(f"Final C2 centroid = {C2_centroid}", file=f)

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

def K_means(data, K, N):
   C1_centroid, C2_centroid = set_initial_centroids(data, K)

   for n in range(0, N, 1): #complete at most N iterations
      new_C1_centroid = [0, 0, 0]
      new_C2_centroid = [0, 0, 0]
      label = []
      for i in range(1, len(data) + 1, 1):
         
         C1_distance = distance_calculation(C1_centroid, data.loc[i])
         C2_distance = distance_calculation(C2_centroid, data.loc[i])
         if (C1_distance > C2_distance):
            label.append("C2")
            #keep running total of sums for each coordinate, to calculate means for next centroid
            new_C2_centroid[0] = new_C2_centroid[0] + data.loc[i]["A"] 
            new_C2_centroid[1] = new_C2_centroid[1] + data.loc[i]["B"] 
            new_C2_centroid[2] = new_C2_centroid[2] + data.loc[i]["C"] 
         else: 
            label.append("C1")
            #keep running total of sums for each coordinate, to calculate means for next centroid
            new_C1_centroid[0] = new_C1_centroid[0] + data.loc[i]["A"] 
            new_C1_centroid[1] = new_C1_centroid[1] + data.loc[i]["B"] 
            new_C1_centroid[2] = new_C1_centroid[2] + data.loc[i]["C"] 
      data["Label"] = label

      new_C1_centroid = [ i / len(data[data["Label"] == "C1"]) for i in new_C1_centroid]
      new_C2_centroid = [ i / len(data[data["Label"] == "C2"]) for i in new_C2_centroid]

      if (new_C1_centroid == C1_centroid) and (new_C2_centroid == C2_centroid):
         break
      else:
         C1_centroid = new_C1_centroid
         C2_centroid = new_C2_centroid
   return data, C1_centroid, C2_centroid, n



def set_initial_centroids(data, K):
   data_size = len(data)
   centroid_indices = random.sample(range(1, (data_size + 1), 1), K) #selects 2 random indices for centroids
   C1_centroid = [data.loc[centroid_indices[0]]["A"], data.loc[centroid_indices[0]]["B"], data.loc[centroid_indices[0]]["C"]]
   C2_centroid = [data.loc[centroid_indices[1]]["A"], data.loc[centroid_indices[1]]["B"], data.loc[centroid_indices[1]]["C"]]
   return C1_centroid, C2_centroid

#using Euclidean distance formula:
#EuclideanDistance = sqrt(sum for i in coordinates A, B, C (centroid[i] – point[i])^2)
def distance_calculation(centroid, point):
   sum = (float(centroid[0]) - float(point["A"]))**2
   sum = sum + (float(centroid[1]) - float(point["B"]))**2
   sum = sum + (float(centroid[2]) - float(point["C"]))**2
   euclideanDistance = math.sqrt(sum)
   return euclideanDistance


if __name__ == "__main__":
   main()
