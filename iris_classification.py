
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame from the dataset
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Map target values to flower names
df['flower_names'] = df.target.apply(lambda x: iris.target_names[x])

# Split the dataset for different flower types
df0 = df[:50]  # Setosa
df1 = df[50:100]  # Versicolor
df2 = df[100:]  # Virginica

# Scatter plots for visualization
plt.figure(figsize=(8,6))
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='green', marker='+', label='Setosa')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='red', marker='.', label='Versicolor')
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='green', marker='+', label='Setosa')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='blue', marker='.', label='Versicolor')
plt.legend()
plt.show()

# Split the dataset into training and test sets
x = df.drop(['target', 'flower_names'], axis='columns')
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train the SVM model
model = SVC()
model.fit(x_train, y_train)

# Make a prediction
print("Prediction for [4.8, 3.0, 1.4, 0.3]:", model.predict([[4.8, 3.0, 1.4, 0.3]]))

# Model accuracy
print("Model accuracy:", model.score(x_test, y_test))
