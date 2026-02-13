
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


# Column 4: Name

train_ds["Name"].head()
len(train_ds["Name"].unique())
# The variable contains the names of the passengers.
# No name is repeated.
# How many repeated surnames in the variable?
surnames = []
for i in range(0, train_ds.shape[0]):
    surnames.append(train_ds["Name"][i].split()[0].split(',')[0])

surnames = pd.Series(surnames)
len(surnames.unique())
# Only 661 surnames are unique, so some of them are repeated.
# Let's create a dataframe with two columns, one with the passenger id and other
# with the passenger surname.
repeated_df = pd.DataFrame({"PassengerId" : train_ds["PassengerId"], "surname" : surnames})

repeated = []
for i in repeated_df["surname"]:
    repeated.append(list(repeated_df[repeated_df["surname"] == i].index.values))


repeated_df = repeated_df.assign(repeated = repeated)
len_rep = []
more_than_one = []
for i in range(0, repeated_df.shape[0]):
    more_than_one.append(len(repeated_df["repeated"][i]) > 1)
    len_rep.append(len(repeated_df["repeated"][i]))

repeated_df = repeated_df.assign(len_rep = len_rep, more_than_one = more_than_one)

# Let's inspect the first repeated surnames
train_ds.loc[repeated_df["repeated"][0]][["Name", "Pclass"]]
# It seems that in this case the passengers are brothers or maye father and son.
train_ds.loc[repeated_df["repeated"][0]][["Name", "Pclass", "Age"]]
# Inspecting also their ages, we can reject the father and son hypothesis.
# Let's test another
train_ds.loc[repeated_df["repeated"][3]][["Name", "Pclass", "Age"]]
# Here the case is clearly of a married couple.
# Let's inspect the surname with more appearances
np.sort(repeated_df["len_rep"].unique())
# The surname with more appearaces has up to nine.
repeated_df[repeated_df["len_rep"] == 9]["surname"]
train_ds.iloc[repeated_df.loc[13]["repeated"]][["Name", "Pclass", "Age"]]
# We could thought that all are members of a same family, maybe husband, wife, 
# their children and a husband's brother.

# Column 5: Sex

train_ds["Sex"].head()
len(train_ds["Sex"].unique())
sum(train_ds["Sex"].isnull())

train_ds["Sex"].value_counts()
train_ds["Sex"].value_counts().plot.bar()
percentages = train_ds["Sex"].value_counts()/sum(train_ds["Sex"].value_counts())
percentages = round(100*percentages, 2)
percentages = list(map(lambda x : str(x) + "%", percentages))

counts = train_ds["Sex"].value_counts()
plt.bar(counts.index, counts.values, color = "powderblue")
plt.text(0, counts[0]//2, 
         percentages[0], ha = "center", color = "black")
plt.text(1, counts[1]//2, 
         percentages[1], ha = "center", color = "black")
plt.title("Passengers by sex")
# There were almost twice as many men as women among the passengers.
#
# Now let's check the quantity of survivors for both sexes.
# After that, let's see the same for each of the classes.
survived = [train_ds[train_ds["Sex"] == "male"]["Survived"].value_counts()[1]]
survived.extend([train_ds[train_ds["Sex"] == "female"]["Survived"].value_counts()[0]])
not_survived = [train_ds[train_ds["Sex"] == "male"]["Survived"].value_counts()[0]]
not_survived.extend([train_ds[train_ds["Sex"] == "female"]["Survived"].value_counts()[1]])
# Create a grouped barplot with this data.
counts_frame = pd.DataFrame({"sex" : ["male", "female"], "survived" : survived, 
                             "not_survived" : not_survived})

counts_frame.pivot_table(index = "sex").plot.barh(color = ["darkolivegreen", "greenyellow"], xlabel = "counts", title = "Survivors for each sex")
# Note: add percentages to this dataframe
percentage_yes = [round(100*counts_frame["survived"][0]/sum(counts_frame.loc[0][["survived", "not_survived"]]), 2)]
percentage_yes.extend([round(100*counts_frame["survived"][1]/sum(counts_frame.loc[1][["survived", "not_survived"]]), 2)])
percentage_no = [round(100*counts_frame["not_survived"][0]/sum(counts_frame.loc[0][["survived", "not_survived"]]), 2)]
percentage_no.extend([round(100*counts_frame["not_survived"][1]/sum(counts_frame.loc[1][["survived", "not_survived"]]), 2)])
percentage_yes = list(map(lambda x : str(x) + "%", percentage_yes))
percentage_no = list(map(lambda x : str(x) + "%", percentage_no))
counts_frame = counts_frame.assign(percentage_yes = percentage_yes)
counts_frame = counts_frame.assign(percentage_no = percentage_no)

# Let's observe the survivors by sex and class.

classes = []
for i in range(1, 4):
    classes.extend([i]*4)
    
    

# rows = (train_ds["Pclass"] == 1) and (train_ds["Sex"] == "male")
# survived = train_ds[([train_ds["Pclass"] == 1] and [train_ds["Sex"] == "male"])[0]][["Survived", "Sex", "Pclass"]]
# survived = train_ds[([train_ds["Pclass"] == 1] and [train_ds["Sex"] == "male"])[0]][["Survived", "Sex", "Pclass"]]
sex = ["male"]*2
sex.extend(["female"]*2)
cs_frame = pd.DataFrame({"class" : classes, "sex" : sex*3, "survived" : [0, 1]*6})
# Note: when the dataframe is completed, add a column with values male1, female1, 
# male2, female2 ... It will substitute the values of the column sex in the 
# previous dataframe.

# Male, class 1
condition1 = train_ds["Pclass"] == 1
condition2 = train_ds["Sex"] == "male"
condition1 = [condition1]
condition2 = [condition2]
# test1 = (condition1 and condition2)[0]
selection = []
for i in range(0, len(condition1[0])):
    selection.append(condition1[0][i] and condition2[0][i])
    

counts = list(train_ds.loc[selection][["Sex", "Pclass", "Survived"]].value_counts().values)

# Female, class 1
condition1 = train_ds["Pclass"] == 1
condition2 = train_ds["Sex"] == "female"
condition1 = [condition1]
condition2 = [condition2]
selection = []
for i in range(0, len(condition1[0])):
    selection.append(condition1[0][i] and condition2[0][i])
    

counts.extend(list(train_ds.loc[selection][["Sex", "Pclass", "Survived"]].value_counts().sort_values().values))

# Male, class 2
condition1 = train_ds["Pclass"] == 2
condition2 = train_ds["Sex"] == "male"
condition1 = [condition1]
condition2 = [condition2]
selection = []
for i in range(0, len(condition1[0])):
    selection.append(condition1[0][i] and condition2[0][i])
    

counts.extend(list(train_ds.loc[selection][["Sex", "Pclass", "Survived"]].value_counts().values))

# Female, class 2
condition1 = train_ds["Pclass"] == 2
condition2 = train_ds["Sex"] == "female"
condition1 = [condition1]
condition2 = [condition2]
selection = []
for i in range(0, len(condition1[0])):
    selection.append(condition1[0][i] and condition2[0][i])
    

counts.extend(list(train_ds.loc[selection][["Sex", "Pclass", "Survived"]].value_counts().sort_values().values))

# Male, class 3
condition1 = train_ds["Pclass"] == 3
condition2 = train_ds["Sex"] == "male"
condition1 = [condition1]
condition2 = [condition2]
selection = []
for i in range(0, len(condition1[0])):
    selection.append(condition1[0][i] and condition2[0][i])
    

counts.extend(list(train_ds.loc[selection][["Sex", "Pclass", "Survived"]].value_counts().values))

# Female, class 3
condition1 = train_ds["Pclass"] == 3
condition2 = train_ds["Sex"] == "female"
condition1 = [condition1]
condition2 = [condition2]
selection = []
for i in range(0, len(condition1[0])):
    selection.append(condition1[0][i] and condition2[0][i])
    

counts.extend(list(train_ds.loc[selection][["Sex", "Pclass", "Survived"]].value_counts().values))

cs_frame = cs_frame.assign(counts = counts)

# Adding a new column that sintetizes the information about the sex and the class
sexclass = []
for i in range(0, cs_frame.shape[0]):
    sexclass.append(cs_frame.iloc[i]["sex"] + str(cs_frame.iloc[i]["class"]))

cs_frame = cs_frame.assign(sexclass = sexclass)

# Note: try to pivot table by not selecting the sex column and by changing sur-
# vived by an str version of itself.
# First let's try to convert survived column to string, if this doesn't works,
# let's try again with a new sexclass column with the form 1male0 for a male
# of the class 1 that survived the crash.
survived2 = list(map(lambda x : str(x), cs_frame["survived"].values))
cs_frame = cs_frame.assign(survived2 = survived2)
cs_frame[["counts", "survived2", "sexclass"]].pivot_table(index = "sexclass")

# This didn't worked, let's continue with option two
ssclass = []
for i in range(0, cs_frame.shape[0]):
    ssclass.append(str(cs_frame.iloc[i]["class"]) + cs_frame.iloc[i]["sex"] + str(cs_frame.iloc[i]["survived"]))


cs_frame = cs_frame.assign(ssclass = ssclass)
cs_frame[["ssclass", "counts"]].pivot_table(index = "ssclass").plot.barh(color = ["darkgreen", "green", "springgreen", "mediumspringgreen", "navy", "darkblue", "aquamarine", "turquoise", "brown", "firebrick", "coral", "orangered"])

# Grouped plot example attempt.
###############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.DataFrame()
df['Datetime'] = pd.date_range(start = '01/07/2018', end = '13/08/2021', freq = '15min')
df['Quantity'] = np.random.rand(len(df))
df['month'] = df['Datetime'].dt.month
df['year'] = df['Datetime'].dt.year

df = df.groupby(by = ['month', 'year'])['Quantity'].sum().reset_index()


fig, ax = plt.subplots()

sns.barplot(ax = ax, data = df, x = 'month', y = 'Quantity', hue = 'year')

plt.show()
###############################################################################

class2 = list(map(lambda x : str(x), cs_frame["class"]))
cs_frame = cs_frame.assign(class2 = class2)

fig, ax = plt.subplots()

sns.barplot(ax = ax, data = cs_frame, x = 'class2', y = 'counts', hue = 'sex')

plt.show()







