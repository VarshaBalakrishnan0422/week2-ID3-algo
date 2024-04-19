import streamlit as st
import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Value if node is leaf

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.num_classes = len(np.unique(y))
        self.num_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(value=predicted_class)

        # Stopping criteria
        if depth < self.max_depth:
            best_split = self._find_best_split(X, y)
            if best_split is not None:
                feature_index, threshold = best_split
                indices_left = X[:, feature_index] < threshold
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node = Node(feature_index=feature_index, threshold=threshold,
                            left=self._grow_tree(X_left, y_left, depth + 1),
                            right=self._grow_tree(X_right, y_right, depth + 1))
        return node

    def _find_best_split(self, X, y):
        best_gini = 1
        best_split = None
        for feature_index in range(self.num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                indices_left = X[:, feature_index] < threshold
                gini = self._gini_impurity(y[indices_left], y[~indices_left])
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_index, threshold)
        return best_split

    def _gini_impurity(self, y_left, y_right):
        p_left = len(y_left) / (len(y_left) + len(y_right))
        p_right = len(y_right) / (len(y_left) + len(y_right))
        gini = 1.0 - p_left**2 - p_right**2
        return gini

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] < node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)

    def visualize(self):
        self._visualize_tree(self.tree)

    def _visualize_tree(self, node, depth=0):
        if node is None:
            return

        indent = "  " * depth
        if node.value is not None:
            st.write(indent + "Predicted class:", node.value)
        else:
            st.write(indent + "Feature:", node.feature_index, "Threshold:", node.threshold)
            self._visualize_tree(node.left, depth + 1)
            self._visualize_tree(node.right, depth + 1)

# Streamlit app
st.title("Decision Tree Classifier Demo")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Display dataset
    st.write("Uploaded Dataset:")
    st.write(df)

    # Extract features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Create Decision Tree classifier
    clf = DecisionTree(max_depth=3)
    clf.fit(X, y)

    # Sidebar for input sample
    st.sidebar.header("Input Sample")
    sample = []
    for i in range(df.shape[1] - 1):
        sample.append(st.sidebar.slider(f"Feature {i+1}", float(df.iloc[:, i].min()), float(df.iloc[:, i].max()), float(df.iloc[:, i].mean())))

    sample = np.array(sample)

    # Sidebar button for classification
    st.sidebar.header("Sample Classification")
    if st.sidebar.button("Classify"):
        prediction = clf.predict(sample.reshape(1, -1))
        st.sidebar.success(f"The sample belongs to class {prediction[0]}.")

    # Display decision tree visualization
    st.header("Decision Tree Visualization")
    clf.visualize()
