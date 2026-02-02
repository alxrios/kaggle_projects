
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

# Column 3: Pclass
train_ds["Pclass"].head()
train_ds["Pclass"].unique()
sum(train_ds["Pclass"].isnull())

counts = train_ds["Pclass"].value_counts()
classes = counts.index
percentage = list(map(lambda x : str(x) + "%", round(100*counts/sum(counts), 2).values))
# counts_frame = pd.DataFrame({"survived" : survived, "counts" : counts.values, 
#                              "percentage" : percentage})

counts_frame = pd.DataFrame({"class" : list(map(lambda x : str(x), classes)), 
                             "counts" : counts.values, "percentage" : percentage})


plt.bar(counts_frame["class"], counts_frame["counts"], color = "teal")
plt.text(0, counts_frame["counts"][0]//2, 
         counts_frame["percentage"][0], ha = "center", color = "white")
plt.text(1, counts_frame["counts"][1]//2, 
         counts_frame["percentage"][1], ha = "center", color = "white")
plt.text(2, counts_frame["counts"][2]//2, 
         counts_frame["percentage"][2], ha = "center", color = "white")
plt.title("Class frequencies")

# Percentage of survived for each class
train_ds[train_ds["Pclass"] == 3]["Survived"].value_counts()
train_ds[train_ds["Pclass"] == 1]["Survived"].value_counts()
train_ds[train_ds["Pclass"] == 2]["Survived"].value_counts()

# counts_frame["not_survived"] = pd.Series(np.array([0]*3))
# counts_frame["survived"] = pd.Series(np.array([0]*3))
counts_frame.loc[0, ["not_survived", "survived"]] = train_ds[train_ds["Pclass"] == 3]["Survived"].value_counts().values
counts_frame.loc[1, ["not_survived", "survived"]] = train_ds[train_ds["Pclass"] == 1]["Survived"].value_counts().values
counts_frame.loc[2, ["not_survived", "survived"]] = train_ds[train_ds["Pclass"] == 2]["Survived"].value_counts().values

test = [372, 119]
test = np.array(test)
test = (100*test/sum(test)).round(2)
f = lambda x : list(map(str, x))
ff = lambda x : list(map(lambda y : y + "%", x))
ff(f(test))

(counts_frame.loc[0, ["not_survived", "survived"]]/sum(counts_frame.loc[0, ["not_survived", "survived"]])).values
test = (counts_frame.loc[0, ["not_survived", "survived"]]/sum(counts_frame.loc[0, ["not_survived", "survived"]])).values
for i in range(0, len(test)):
    print(i)

# How to add new columns to an existing dataframe...
counts_frame.assign(new = np.random.rand(3))

new_columns = []
for i in range(0, 3):
    aux = (counts_frame.loc[i, ["not_survived", "survived"]]/sum(counts_frame.loc[i, ["not_survived", "survived"]])).values
    aux = np.array(list(100*aux)).round(2)
    aux = ff(f(aux))
    new_columns.append(aux)


counts_frame = counts_frame.assign(percentage_not = [""]*3, percentage_yes = [""]*3)
for i in range(0, 3):
    aux = (counts_frame.loc[i, ["not_survived", "survived"]]/sum(counts_frame.loc[i, ["not_survived", "survived"]])).values
    aux = np.array(list(100*aux)).round(2)
    aux = ff(f(aux))
    counts_frame.loc[i, ["percentage_not", "percentage_yes"]] = aux


plt.bar(["0", "1"], counts_frame[counts_frame["class"] == "3"][["not_survived", "survived"]], color = "teal")
# This does not work, so let's try with "subframes"
#
# First test...
test = counts_frame[counts_frame["class"] == "3"][["not_survived", "survived"]]
subframe = pd.DataFrame({"survived" : ["0", "1"], "counts" : test.values[0]})
plt.bar(subframe["survived"], subframe["counts"])

# Now with a loop...
for i in counts_frame["class"]:
    counts = counts_frame[counts_frame["class"] == i][["not_survived", "survived"]]
    subframe = pd.DataFrame({"survived" : ["0", "1"], "counts" : counts.values[0]})
    plt.bar(subframe["survived"], subframe["counts"])

# Let's try to make the plots separated
# Class 3
counts = counts_frame[counts_frame["class"] == "3"][["not_survived", "survived"]]
percentages = counts_frame[counts_frame["class"] == "3"][["percentage_not", "percentage_yes"]]
subframe = pd.DataFrame({"survived" : ["0", "1"], "counts" : counts.values[0], "percentage" : percentages.values[0]})
plt.bar(subframe["survived"], subframe["counts"], color = "darkslategray")
plt.text(0, subframe["counts"][0]//2, 
         subframe["percentage"][0], ha = "center", color = "white")
plt.text(1, subframe["counts"][1]//2, 
         subframe["percentage"][1], ha = "center", color = "white")
plt.title("Survived  class 3")
# Class 2
counts = counts_frame[counts_frame["class"] == "2"][["not_survived", "survived"]]
percentages = counts_frame[counts_frame["class"] == "2"][["percentage_not", "percentage_yes"]]
subframe = pd.DataFrame({"survived" : ["0", "1"], "counts" : counts.values[0], "percentage" : percentages.values[0]})
plt.bar(subframe["survived"], subframe["counts"], color = "teal")
plt.text(0, subframe["counts"][0]//2, 
         subframe["percentage"][0], ha = "center", color = "white")
plt.text(1, subframe["counts"][1]//2, 
         subframe["percentage"][1], ha = "center", color = "white")
plt.title("Survived  class 2")
# Class 1
counts = counts_frame[counts_frame["class"] == "1"][["not_survived", "survived"]]
percentages = counts_frame[counts_frame["class"] == "1"][["percentage_not", "percentage_yes"]]
subframe = pd.DataFrame({"survived" : ["0", "1"], "counts" : counts.values[0], "percentage" : percentages.values[0]})
plt.bar(subframe["survived"], subframe["counts"], color = "darkturquoise")
plt.text(0, subframe["counts"][0]//2, 
         subframe["percentage"][0], ha = "center", color = "white")
plt.text(1, subframe["counts"][1]//2, 
         subframe["percentage"][1], ha = "center", color = "white")
plt.title("Survived  class 1")

# Grouped plots
classes = [3]*2
classes.extend([1]*2)
classes.extend([2]*2)
survived = list(counts_frame[counts_frame["class"] == "3"][["not_survived", "survived"]].values[0])
percentage = list(counts_frame[counts_frame["class"] == "3"][["percentage_not", "percentage_yes"]].values[0])
for i in ["1", "2"]:
    survived.extend(list(counts_frame[counts_frame["class"] == i][["not_survived", "survived"]].values[0]))
    percentage.extend(list(counts_frame[counts_frame["class"] == i][["percentage_not", "percentage_yes"]].values[0]))



counts_frame_long = pd.DataFrame({"class" : classes, "survived" : [0, 1]*3, 
                                  "counts" : survived, "percentage" : percentage})


counts_frame[["class", "not_survived", "survived"]].pivot_table(index = "class").plot.barh(title = "Survived for each class", xlabel = "counts", color = ["darkslategray", "darkcyan"])
























