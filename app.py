# Import libraries

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

## APP Heading 
st.write('''
# Explore Different ML Models and datasets
which one is best lets see
''')

##  Select Dataset names
dataset_name = st.sidebar.selectbox("select Dataset",("Iris","Breast Cancer","Wine"))

# Select classifier name
classifier_name = st.sidebar.selectbox("select classifier",("KNN","SVM","Random forest"))

# Define function
def get_dataset (dataset_name):
    data = None
    if dataset_name=="Iris":
        data=datasets.load_iris()
    elif dataset_name=="wine":
        datasizemask=datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    x = data.data
    y = data.target
    return x,y

# ab is function ko x,y variables k equal rakh lain gy
x,y = get_dataset(dataset_name)

# now we will print shape of datset on my app
st.write("Shape of dataset:",x.shape)
st.write("number of classes:",len(np.unique(y)))

# next hum different classifier k parametrs ko user input ma add karein gy
def add_parameter_ui(classifier_name):
    params = dict()    # create an empty dictionary
    if classifier_name=="SVM":
        C = st.sidebar.slider("C",0.01,10.0)
        params["C"] = C  # its the degree of correct classification
    elif classifier_name=="KNN":
        K = st.sidebar.slider("K",1,15)
        params["K"] = K      # its number of nearest neighbors
    else:
        max_depth = st.sidebar.slider("max_depth",2,15)
        params["max_depth"] = max_depth # depth of every that grow in random forest
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        params["n_estimators"] = n_estimators # number of tress
    return params

# hum is function ko params k equal rakh lain gy
params = add_parameter_ui(classifier_name)

# now we will make classifier on classifier name and params
def get_classifier(classifier_name,params):
    clf = None
    if classifier_name =="SVM":
        clf = SVC(C=params["C"])
    elif classifier_name =="KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    else:
        clf = clf=RandomForestClassifier(n_estimators=params["n_estimators"],
        max_depth = params["max_depth"],random_state=1234)
    return clf

## is function ko clf variables k equal rakh lain gy
clf = get_classifier(classifier_name,params)

# now we are doing our dataset split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1234)

# ab hum ny apny classifier ki training krwani 
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

# ab humny accuracy score lna h
acc = accuracy_score(y_test,y_pred)
st.write(f"Classifier={classifier_name}")
st.write(f"Accuracy=",acc)

# ab hum apny sary features ko 2 dimensional plot par drwa karein gy
pca = PCA(2)
x_projected = pca.fit_transform(x)

# ab hum apny data 0 or 1 par dimension par slice krdain gy
x1 = x_projected[:,0]
x2 = x_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis") # cmap=colormap

plt.xlabel("Principle component 1")
plt.ylabel("Principle Component 2")
plt.colorbar()

#plt.show
st.pyplot(fig)


    



    






