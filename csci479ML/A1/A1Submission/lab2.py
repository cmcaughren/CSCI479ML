"""
   Carolyn McAughren Lab2 CSCI 479 Machine Learning

 * Design and implement a program to find the "best" split threshold value
   to perform the two-way split for the "Measurement" attribute based on the GINI information.

 * Pre-process the data and transform the "Measurement" attribute to a categorical one based
   on the split value found in the previous step.
"""
import sys
import pandas as pd
import numpy as np

#program is hardcoded to split "Measurement" data into "2" bins only
#splitting other named categories, or splitting into more than 2 bins will require reworking 
#this program 

print_state_bool = True

def main():
   if len(sys.argv) is not 2:
      print(f"ERROR: Expecting 1 command line arguements: a filename containing the data to be split. \n \
         ex. 'python3 lab1.py A1-data.csv' ")
      quit()

   data = read_in_data()
   attribute = confirm_attribute(data)
   updated_data = best_split(data, attribute, print_state_bool)
   #updated_data.to_csv('updated_data.csv', index=False)
   

#confirms the data file exists
def read_in_data():
   try:
      data = pd.read_csv(sys.argv[1])
      return data
   except FileNotFoundError:
      print(f"\nERROR: The file {sys.argv[1]} does not exist.")
      quit()

#check that attribute exists in the data
#return the attribute title or quit with error message if not found
def confirm_attribute(data):
   attribute = "Measurement"
   columns = data.columns.values
   if attribute not in columns:
      print(f"'Measurement' is not a column header in the provided data.")
      quit()
   return attribute

#determine the best value to split Measurement into 2 categories, considering 
#the target value "STATUS" distribution for each potential split 
def best_split(data, attribute, print_state_bool):

   #Step 1.
   #find best Vo value to split Measurement by

   #Step 1a. 
   #create table1 with columns:
   # Measurement | cnt_Measurement | cnt_Norm_Status | cnt_Crit_Status
   # Where: cnt_Measurement is the number of items with that Measurement
   #        cnt_Norm_Status is the number of items with that Measurement and are Status Normal
   #        cnt_Crit_Status is the number of items with that Measurement and are Status Critical


   #count the number of each item with each Measurement value (ie group by Measurement value)
   #create a new dataframe holding these measurement values, and their counts
   table1 = data.groupby(['Measurement']).size()
   table1 = table1.reset_index()
   table1.columns = ["Measurement", "cnt_Measurement"]

   #count the number of items with 'normal' status, grouped by measurement value
   #create a new dataframe holding these values
   cnt_Norm_Status_df = data[data["STATUS"] == 'Normal'].groupby(['Measurement']).size()
   cnt_Norm_Status_df = cnt_Norm_Status_df.reset_index()
   cnt_Norm_Status_df.columns = ["Measurement", "cnt_Norm_Status"]
   cnt_Norm_Status_df = pd.DataFrame(cnt_Norm_Status_df)

   #count the number of items with 'critical' status, grouped by measurement value
   #create a new dataframe holding these values
   cnt_Crit_Status_df = data[data["STATUS"] == 'Critic'].groupby(['Measurement']).size()
   cnt_Crit_Status_df = cnt_Crit_Status_df.reset_index()
   cnt_Crit_Status_df.columns = ["Measurement", "cnt_Crit_Status"]
   cnt_Crit_Status_df = pd.DataFrame(cnt_Crit_Status_df)
   
   #merge the table1 dataframe with the normal status counts dataframe, on Measurement column
   table1 = pd.merge(table1, cnt_Norm_Status_df, on="Measurement", how="left")
   #replace NaN values with 0
   table1 = table1.fillna(0)
   #cast float dtypes (due to NaNs) to ints 
   table1.cnt_Norm_Status = table1.cnt_Norm_Status.astype(int)

   #merge the table1 dataframe with the crit status counts dataframe, on Measurement column
   table1 = pd.merge(table1, cnt_Crit_Status_df, on="Measurement", how="left")
   #replace NaN values with 0
   table1 = table1.fillna(0)
   #cast float dtypes (due to NaNs) to ints 
   table1.cnt_Crit_Status = table1.cnt_Crit_Status.astype(int)

   if print_state_bool:
      print("TABLE 1. Counts of items with Normal and Critical Status, grouped by Measurement value.")
      print(table1.to_string())
      print("\n")

   #Step 1b. 
   #create table2 with columns:
   # split_value | cnt_M1 | cnt_M2 | cnt_N_M1 | cnt_C_M1 | cnt_N_M2 | cnt_C_M2 | Gini
   # Where: split_value is candidate Vo
   #        cnt_M1 is number of items with Measurement below split_value
   #        cnt_M2 is number of items with Measurement equal to or above split_value
   #        cnt_N_M1 is the number of items in M1 with Normal Status
   #        cnt_C_M1 is the number of items in M1 with Critical Status
   #        cnt_N_M2 is the number of items in M2 with Normal Status
   #        cnt_C_M2 is the number of items in M2 with Critical Status
   #        GINI is the calculated gini index for that Vo split_value


   #Test each value in Measurement column of table1 as Vo
   total_items = len(data)
   total_Norm_items = len(data[data["STATUS"] == "Normal"]) 
   total_Crit_items = len(data[data["STATUS"] == "Critic"]) 

   if print_state_bool:
      print(f"Total items: {total_items}, Normal items: {total_Norm_items}, Critical items: {total_Crit_items}")
      print("\n")
   #variables used to remember values from previous rows
   #used to speed up calculations as table2 is built 
   M1 = 0
   M1N = 0
   M1C = 0

   table2_list = []
   for row in table1.itertuples():
      #all items <= split_value go in M1, all >split_value go in M2
      split_value = row.Measurement
      M1 = M1 + row.cnt_Measurement
      M2 = total_items - M1
      M1N = M1N + row.cnt_Norm_Status
      M1C = M1C + row.cnt_Crit_Status
      M2N = total_Norm_items - M1N
      M2C = total_Crit_items - M1C

      if M1 != 0 and M2 != 0 :
         GINI = (M1/total_items)*(1 - (M1N/M1)**2 - (M1C/M1)**2) + (M2/total_items)*(1 - (M2N/M2)**2 - (M2C/M2)**2)
      elif M1 == 0:
         GINI = (1 - (M2N/M2)**2 - (M2C/M2)**2)
      else:
         GINI = (1 - (M1N/M1)**2 - (M1C/M1)**2)

      row_list = [split_value, M1, M2, M1N, M1C, M2N, M2C, GINI]
      table2_list.append(row_list)

   table2 = pd.DataFrame(table2_list)
   table2.columns = ["split_value", "cnt_M1", "cnt_M2", "cnt_N_M1", "cnt_C_M1", "cnt_N_M2", "cnt_C_M2", "GINI"]
   
   if print_state_bool:
      print("TABLE 2. Counts of how items are split around split_value and the resulting GINI for that split.")
      print(table2.to_string())
      print("\n")

   #Step 1c
   #Select best Vo from table2 = the split value with lowest Gini index
   Vo_idx = table2['GINI'].idxmin()
   Vo = table2.iloc[Vo_idx]['split_value']
   
   if print_state_bool:
      print(f"The split value with the lowest GINI is: {Vo}")
      print("\n")

   #Step 2. 
   # Add column for Measurement_Categorical to original data
   # Based on Measurement value and best Vo (calculated above) assign each item:
   # 0 if Measurement is below Vo
   # 1 if Measurement is above or equal to Vo
   data['Measurement_Category'] = np.where(data["Measurement"] <= Vo, 0, 1)
   
   if print_state_bool:
      print("FINAL TABLE: Including new Measurement_Category column and resulting values.")
      print(data.to_string())
      print("\n")

   #return new list with this added attribute
   return data



if __name__ == "__main__":
   main()