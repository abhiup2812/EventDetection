#merging all the different disaster classes into one CSV file for multiclass labeling
import numpy as np
import pandas as pd

df1 =pd.read_csv('2012_Sandy_Hurricane-ontopic_offtopic.csv', 
            sep = r',', 
            skipinitialspace = True)

df1["label"] = np.where(df1["label"]=="off-topic", "NONE", "Hurricane")

df1.head()

df2 =pd.read_csv("2013_Alberta_Floods-ontopic_offtopic.csv", 
            sep = r',', 
            skipinitialspace = True)

df2["label"] = np.where(df2["label"]=="off-topic", "NONE", "Alb_flood")

df3 =pd.read_csv("2013_Boston_Bombings-ontopic_offtopic.csv", 
            sep = r',', 
            skipinitialspace = True)

df3["label"] = np.where(df3["label"]=="off-topic", "NONE", "Bombing")

df4 =pd.read_csv("2013_Oklahoma_Tornado-ontopic_offtopic.csv", 
            sep = r',', 
            skipinitialspace = True)

df4["label"] = np.where(df4["label"]=="off-topic", "NONE", "Torndo")

df5 =pd.read_csv("2013_Queensland_Floods-ontopic_offtopic.csv", 
            sep = r',', 
            skipinitialspace = True)

df5["label"] = np.where(df5["label"]=="off-topic", "NONE", "Queens_flood")

df6 =pd.read_csv("2013_West_Texas_Explosion-ontopic_offtopic.csv", 
            sep = r',', 
            skipinitialspace = True)

df6["label"] = np.where(df6["label"]=="off-topic", "NONE", "Explosion")

comb_data = [df1,df2,df3,df4,df5,df6]

new_data = pd.concat(comb_data)

new_data.shape

new_data.drop_duplicates(keep=False, inplace=True)

new_data.shape

from sklearn.utils import shuffle
new_data1 = shuffle(new_data)

new_data1.head()

new_data1.shape

new_data1.to_csv("CrisisLexT6.csv", sep=',', encoding='utf-8')