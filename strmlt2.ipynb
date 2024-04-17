import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Define function to predict class of new sample
def predict_class(sample):
    prediction = clf.predict([sample])
    return iris.target_names[prediction[0]]

# Streamlit app
st.title('Decision Tree Classifier')
st.sidebar.header('Input Sample')

# Input fields for user to input new sample
sepal_length = st.sidebar.slider('Sepal Length', float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.sidebar.slider('Sepal Width', float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.sidebar.slider('Petal Length', float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.sidebar.slider('Petal Width', float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Show decision tree visualization
st.subheader('Decision Tree Visualization')
fig, ax = plt.subplots(figsize=(10, 7))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, ax=ax)
st.pyplot(fig)

# Predict class of new sample
sample = [sepal_length, sepal_width, petal_length, petal_width]
predicted_class = predict_class(sample)

# Display prediction
st.subheader('Prediction')
st.write('The predicted class for the given sample is:', predicted_class)