
"""
The objective of this script is to analize the titanic dataset.
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

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

# Column 2: Survived
train_ds["Survived"].head()
train_ds["Survived"].unique()
sum(train_ds["Survived"].isnull())
# It consist of two values, 0 if the passanger didn't survived and 1 if he did it.
# This will be the variable that we want to predict.
counts = train_ds["Survived"].value_counts()
survived = counts.index
percentage = list(map(lambda x : str(x) + "%", round(100*counts/sum(counts), 2).values))
# counts_frame = pd.DataFrame({"survived" : survived, "counts" : counts.values, 
#                              "percentage" : percentage})

counts_frame = pd.DataFrame({"survived" : list(map(lambda x : str(x), survived)), 
                             "counts" : counts.values, "percentage" : percentage})

print(counts_frame)
print(counts_frame.to_string(index = False))

plt.bar(counts_frame["survived"], counts_frame["counts"], color = "seagreen")
plt.title("Survived frequencies")

# Trying to add labels to the bars
fig, ax = plt.subplots()
bar_container = ax.bar(counts_frame["survived"], counts_frame["counts"])
ax.set(ylabel='speed in MPH', title='Running speeds')
ax.bar_label(bar_container, fmt='{:,.0f}')
# Another try...
fig, ax = plt.subplots()
bar_container = ax.bar(counts_frame["survived"], counts_frame["counts"])
ax.set(ylabel='speed in MPH', title='Running speeds')
ax.set_yticks(counts_frame["counts"], labels = counts_frame["percentage"])
# Third try...
fruit_names = ['Coffee', 'Salted Caramel', 'Pistachio']
fruit_counts = [4000, 2000, 7000]

fig, ax = plt.subplots()
bar_container = ax.bar(counts_frame["survived"], counts_frame["counts"])
ax.set(ylabel='pints sold', title='Gelato sales by flavor')
ax.bar_label(bar_container, fmt='{:,.0f}')

f = lambda x: f'{x :.2f}%'
f(61.62)

# Let's do again the counts_frame dataframe
percentage = round(100*counts/sum(counts), 2).values
counts_frame = pd.DataFrame({"survived" : list(map(lambda x : str(x), survived)), 
                             "counts" : counts.values, "percentage" : percentage})

# Another try...
plt.bar(counts_frame["survived"], counts_frame["counts"])
plt.text(0, counts_frame["counts"][0] - 200, 
         counts_frame["percentage"][0])


# Testing a new code...
# Function to add value labels on top of bars
def add_labels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i] - 1000, y[i])  # Placing text slightly above the bar

# Data for the bar chart
x = ["Engineering", "BSc", "MBA", "Bcom", "BBA", "MSc"]
y = [9330, 4050, 3030, 5500, 8040, 4560]

# Creating bar chart
plt.bar(x, y)

# Adding value labels
add_labels(x, y)

# Adding title and labels
plt.title("College Admission")
plt.xlabel("Courses")
plt.ylabel("Number of Admissions")

# Display the chart
plt.show()

# Another try...
plt.bar(counts_frame["survived"], counts_frame["counts"], color = "seagreen")
plt.text(0, counts_frame["counts"][0] - 340, 
         counts_frame["percentage"][0], ha = "center")
plt.text(1, counts_frame["counts"][1] - 150, 
         counts_frame["percentage"][1], ha = "center")
plt.title("Survived frequencies")

# Using half of the height
plt.bar(counts_frame["survived"], counts_frame["counts"], color = "seagreen")
plt.text(0, counts_frame["counts"][0]//2, 
         counts_frame["percentage"][0], ha = "center", color = "white")
plt.text(1, counts_frame["counts"][1]//2, 
         counts_frame["percentage"][1], ha = "center", color = "white")
plt.title("Survived frequencies")




