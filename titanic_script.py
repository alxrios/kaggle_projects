
"""
The objective of this script is to analize the titanic dataset.
"""

import pandas as pd
import os

# Data load
os.chdir("./documents/datasets/titanic")

train_ds = pd.read_csv("train.csv")
train_ds.shape
# The dataset has 891 rows and 12 columns. Let's start doing an exploratory 
# data analysis of the columns.
train_ds.columns.values

# Column 1: PassengerId
train_ds["PassengerId"].values[0:10]
type(train_ds["PassengerId"].values[0])
sum(train_ds["PassengerId"].isnull())
# The column does not contain null values.
len(train_ds["PassengerId"].unique())
# So the column contains a unique id for each observation.
[train_ds["PassengerId"].min(), train_ds["PassengerId"].max()]
# Are the ids ordered?
sum(train_ds["PassengerId"].sort_values() == train_ds["PassengerId"])
# Ans: yes
# So, this variable only identifies each observation and will not be usefull
# for prediction purposes.



































