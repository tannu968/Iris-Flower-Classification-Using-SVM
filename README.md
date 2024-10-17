# Iris-Flower-Classification-Using-SVM

This project focuses on classifying Iris flowers based on their features using the Support Vector Machine (SVM) model. The dataset used is the famous Iris dataset, which consists of three types of flowers: Setosa, Versicolor, and Virginica.

Project Overview
Load and preprocess the Iris dataset.
Visualize data using scatter plots.
Train an SVM model to classify the flowers based on their features.
Evaluate the model's accuracy on the test data.

Key Libraries Used
Pandas
Matplotlib
Scikit-learn

Dataset
The Iris dataset contains the following features:

Sepal Length (cm)
Sepal Width (cm)
Petal Length (cm)
Petal Width (cm)

The dataset is split into training and testing sets, and a Support Vector Machine (SVM) model is trained to classify the target variable.

Visualization
Scatter plots are used to visualize the relationship between different features:

Sepal length vs. Sepal width
Petal length vs. Petal width

Model
We use the Support Vector Machine (SVC) from Scikit-learn for classification. The model is trained on 80% of the dataset, and the remaining 20% is used for testing.

How to Run the Project

git clone https://github.com/your-username/iris-flower-classification.git

pip install -r requirements.txt

python iris_classification.py

Results
The model is evaluated based on accuracy, and its performance is compared against test data.

Future Enhancements
Apply different classification models like Decision Tree or Random Forest for comparison.
Perform hyperparameter tuning for better accuracy.
