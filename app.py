import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

# Sidebar main app
# ------------------------------------------------------------------------------
st.sidebar.header('1. Main App Settings')
st.sidebar.markdown("""
Change some of the settings down below and see how the chosen dataset and algorithm will be affected.
""")
dataset = st.sidebar.selectbox(
    "Select Dataset", ("Iris", "Breast Cancer", "Wine"))
classifer = st.sidebar.selectbox(
    "Select Algorithm", ("KNN", "SVM", "Random Forest"))


# Body main app
# ------------------------------------------------------------------------------
st.title('1. Streamlit Showcase Main App')
st.subheader('Just a simple showcase to see how streamlit works using the built-in sklearn datasets iris, breast cancer and wine in combination with k-nearest neighbors, support vector machines and random forest classifiers. If not visible open the sidebar on the left.')

st.write("""## Model Properties""")
st.write("**Progress**")
progress = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.05)
    progress.progress(percent_complete+1)

st.write("**Selected dataset: **" + dataset.upper() +
         " / **Selected classifier: **" + classifer.upper())

# Get the values of the dataset chosen by the user
def get_dataset(dataset):
    if dataset == "Iris":
        data = datasets.load_iris()
    elif dataset == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y


# Provide information about the chosen dataset
X, y = get_dataset(dataset)
st.write("**Shape of dataset:**", X.shape)
st.write("**Number of classes:**", len(np.unique(y)))

# Provide parameter sliders depending on the chosen classifier
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
        st.sidebar.markdown(
            "For further KNN parameters see (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)")
    elif clf_name == "SVM":
        C = st.sidebar.slider("C (regularization parameter)", 0.01, 10.0)
        params["C"] = C
        st.sidebar.markdown(
            "For further SVM paramters see (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)")
    else:
        max_depth = st.sidebar.slider("max depth", 2, 15)
        n_estimators = st.sidebar.slider("numer of estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        st.sidebar.markdown(
            "For further random forest parameters see(https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)")
    return params


# Call the parameter sliders function
params = add_parameter_ui(classifer)

# Get the sklearn classfiers as imported above
def get_classifier(classifier, params):
    if classifier == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif classifer == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"], random_state=1234)
    return clf

# Call the get_classifier function
clf = get_classifier(classifer, params)

# Do the classification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Display the model results
st.write("""## Model Results""")
st.write(f"classifier = {clf}")
st.write(f"Accuracy = {acc}")

st.write("""## Principal Component Analysis""")

#Plot the dataset
pca = PCA(n_components=2, svd_solver="auto")
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
#plt.show() not used in this case
#Instead using the streamlit built-in function
st.pyplot(fig)

#pd.DataFrame(datasets.load_iris())

st.write("""
## Tech Stack
:rocket: Python-numpy-pandas-matplotlib-sklearn-Streamlit-Github-Heroku-HTML <iframe>
## TODO
a) Check out caching features in order to avoid the whole script running again (decorators?)\n
b) Display a dataset table\n
c) Add more classifier parameters\n
d) Add other classifiers\n
e) Add feature scaling (0 to 1)
## Credits
Code inspiration by Python Engineer (https://www.youtube.com/watch?v=Klqn--Mu2pE)
Streamlit: https://www.streamlit.io/
""")

# Sidebar additional showcase elements
# ------------------------------------------------------------------------------
st.sidebar.header('2. Additional show case elements')
uploadfile = st.sidebar.file_uploader('Upload your csv-file', type=["csv"])
#dataset_slider = st.sidebar.slider('Length', 0, 100, 50)
#classifier_slider = st.sidebar.slider('Width', 0, 100, 50)

# Body additional showcase elements
# ------------------------------------------------------------------------------

st.write("""___""")
st.title('2. Additional showcase elements')
st.subheader('Random numbers written into a pandas data frame')

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])
st.area_chart(chart_data)

st.subheader("Scatter plot showing four dimensions on the x-axis, y-axis and by bubble size and bubble color")
st.vega_lite_chart(chart_data, {
     'mark': {'type': 'circle', 'tooltip': True},
     'encoding': {
             'x': {'field': 'a', 'type': 'quantitative'},
             'y': {'field': 'b', 'type': 'quantitative'},
             'size': {'field': 'c', 'type': 'quantitative'},
             'color': {'field': 'c', 'type': 'quantitative'},
    },
})

st.subheader('Random longitudes and latitudes shown on a map')

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

# st.balloons()

st.subheader('Code example')

st.write("""
```bash
Streamlit
____________________________
st.write('First data table')
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40],
    'third column': [7, 8, 9, 10]
}))

Markdown emojis
____________________________
:+1: :sparkles: :camel: :tada: :rocket: :metal: :octocat:
```
""")

#if __name__ == "__main__":
#    pass
