import pandas as pd
import matplotlib.pyplot as plt

 # Load the Titanic training dataset
train_data = pd.read_csv("train.csv")   

# Drop rows with missing age values
train_data = train_data.dropna(subset=["Age"])

# Sort the data by PassengerId (time order)
train_data = train_data.sort_values("PassengerId")



# Create a line plot function
def create_line_plot(data, x_col, y_col, title, x_label, y_label):
    plt.figure(figsize=(10, 6))
    plt.plot(data[x_col], data[y_col], marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()

# Create a line plot of passenger ages over time
create_line_plot(
    train_data, 
    x_col="PassengerId", 
    y_col="Age", 
    title="Passenger Ages Over Time", 
    x_label="Passenger ID", 
    y_label="Age"
)

# Drop rows with missing age and fare values
train_data = train_data.dropna(subset=["Age", "Fare"])


# Create a scatter plot function
def create_scatter_plot(data, x_col, y_col, title, x_label, y_label):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[x_col], data[y_col], color='g', alpha=0.5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()


#Create a scatter plot of passenger ages vs. fares
create_scatter_plot(
    train_data, 
    x_col="Age", 
    y_col="Fare", 
    title="Passenger Ages vs. Fares", 
    x_label="Age", 
    y_label="Fare"
)


# Create a bar chart function
def create_bar_chart(data, x_col, y_col, title, x_label, y_label):
    plt.figure(figsize=(10, 6))
    plt.bar(data[x_col], data[y_col], color='r')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()
    
# Create a bar chart of passenger classes
class_counts = train_data["Pclass"].value_counts().reset_index()
class_counts.columns = ["Class", "Count"]
plt.figure(figsize=(10, 6))
plt.bar(class_counts["Class"], class_counts["Count"], color='r')
plt.title("Passenger Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.grid(True)
plt.show()



def create_histogram(data, column, bins=10):
    plt.figure(figsize=(10, 6))
    plt.hist(data[column], bins=bins, edgecolor="k", alpha=0.5, color='blue')
    plt.title(f"{column} Distribution")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(True)

# Calling the function to create a histogram of passenger ages
create_histogram(train_data, "Age", bins=20)
plt.show()



def create_box_plot(data, column):
    plt.figure(figsize=(10, 6))
    plt.boxplot(data[column], vert=False, widths=0.7, patch_artist=True,
                boxprops=dict(facecolor="blue"))
    plt.title(f"{column} Distribution")
    plt.xlabel(column)
    plt.grid(True)

# Calling the function to create a box plot of passenger ages
create_box_plot(train_data, "Age")
plt.show()




def create_pie_chart(data, column):
    value_counts = data[column].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f"{column} Distribution")
    plt.axis("equal")

# Calling the function to create a pie chart of passenger class distribution
create_pie_chart(train_data, "Pclass")
plt.show()




