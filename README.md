# streamlit-showcase
Showcase using streamlit to create a web app showing the effects of sklearn classification algorithms including user interaction
___
## Last Update

2020-11-29 [yyyy-mm-dd]
___
## Tech Stack

:rocket: Python-numpy-pandas-matplotlib-sklearn-Streamlit-Github-Heroku-HTML <iframe>
___
## Python

Python 3.8.5
___
## Requirements

| Package              | Version   
| -------------------- | --------  
| matplotlib           | 3.3.3
| numpy                | 1.19.2
| pandas               | 1.1.4
| scikit-learn         | 0.23.2
| sklearn              | 0.0
| streamlit            | 0.71.0
___
## Issues

:ok:
___
## :construction: TODO

* Check out caching features in order to avoid the whole script running again (decorators?)
* Display a dataset table
* Add other classifiers
* Add feature scaling (0 to 1)
* Add more classifier parameters

* K-nearest neighbors: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
* Support vector machine: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
* Random forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
___
## Content

Streamlit is a framework that allows for using Python to display content in the browser. Rendering the code (HTML, CSS, Javascript) is actually done under the hood.The Github repo can be found here: https://github.com/streamlit.

This simple demo app lets the user to first chose one of the built-in sklearn datasets and then a classfication algorithm. With the sliders one can change some example parameters.

Once the model has been updated based on the user's decisions the following results will be shown:

* selected dataset
* selected classifier
* shape of the dataset (number of records)
* number of unique classes
* accuracy of the respective combination of dataset and classifier

A two-dimensional principal component analysis chart will be shown as well.

Down below in part 2 are some mockup widgets in the slider and some charts will be displayed based on random figures. Just for demonstration purposes.
___
## Deployment

The app was deployed by committing it to Github and then using Heroku as a hosting platform. There are several ways to deploy to Heroku
* Download the Heroku command line interface (CLI) and use heroku commands directly in the terminal
* Use Github to establish a continuous integration / continuous deployment (CI/CD) pipeline to Heroku.

The following files need to be uploaded to Heroku as well in order to make this app work:

* app.py (your python code using streamlit and the other dev and production dependencies)
* Procfile (run the setup shell skript and then the streamlit app)  
* requirements.txt (list of production dependencies)
* setup.sh (shell skript creating a streamlit toml file)

*Steps*
### 1. Create a Github repo and clone it locally on your machine with
```bash
git clone https://github.com/[your_github_name]/[your_app_name].git
```
### 2. cd into your local directory where the git clone with [your_app_name] is located and create the the four files required

a) [your_app.py]
```bash
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

[your Python code]
```
b) The Heroku Procfile (without extensions in the file name) containing the following code:
```bash
web: sh setup.sh && streamlit run [your_app_name].py
```
c) The requirements.txt
```bash
pip freeze > requirements.txt
```
d) The setup shell skript. Use the following code:
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml
```
___
## Sources / Credits

* Code Inspiration by **Python Engineer** https://www.youtube.com/watch?v=Klqn--Mu2pE
* Streamlit: https://www.streamlit.io/

