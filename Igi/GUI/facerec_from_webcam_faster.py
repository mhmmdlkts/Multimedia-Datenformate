import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Visualization
import matplotlib.pyplot as plt

#Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics

#System
import os

#  In this study, face recognition was performed using the face images in the Olivetti data set. The steps for face recognition are as follows:
#
#     Principal components of face images were obtained by PCA.
#     Adequate number of principal components determined
#     According to three different classification models, accuracy score obtained.
#     According to three different classification models, cross-validation accuracy score were obtained.
#     Parameter optimization of the best model has been made.
#
# Go to Contents Menu | Quick Links: 1. |3.|3.1. |3.2.|4.|4.1.|4.2.|4.3.|4.4. 4.5. |4.6. |4.7. |4.8. |4.9. |4.10.|4.11.|4.12.
#
# 2.Face Recognition
#
# The first study on automatic facial recognition systems was performed by Bledsoe between 1964 and 1966. This study was semi-automatic. The feature points on the face are determined manually and placed in the table called RAND. Then, a computer would perform the recognition process by classifying these points. However, a fully functional facial recognition application was performed in 1977 by Kanade. A feature-based approach was proposed in the study. After this date, two-dimensional (2D) face recognition have studied intensively. Three-dimensional (3D) face studies were started to be made after the 2000s.
#
# 3D facial recognition approaches developed in a different way than 2D facial recognition approaches. Therefore, it will be more accurate to categorize in 2D and 3D when discussing face recognition approaches.
#
# We can classify the face recognition researches carried out with 2D approach in three categories; analytical (feature-based, local), global (appearance) and hybrid methods. While analytical approaches want to recognize by comparing the properties of the facial components, global approaches try to achieve a recognition with data derived from all the face. Hybrid approaches, together with local and global approaches, try to obtain data that expresses the face more accurately.
#
# Face recognition performed in this kernel can assessed under global face recognition approaches.
#
# In analytical approaches, the distance of the determined feature points and the angles between them, the shape of the facial features or the variables containing the regional features are obtained from the face image are used in face recognition. Analytical methods examine the face images in two different ways according to the pattern and geometrical properties. In these methods, the face image is represented by smaller size data, so the big data size problem that increases the computation cost in face recognition is solved.
#
# Global-based methods are applied to face recognition by researchers because they perform facial recognition without feature extraction which is troublesome in feature based methods. Globally based methods have been used in face recognition since the 1990s, since they significantly improve facial recognition efficiency. Kirby and Sirovich (1990) first developed a method known as Eigenface, which is used in facial representation and recognition based on Principal Component Analysis . With this method, Turk and Pentland transformed the entire face image into vectors and computed eigenfaces with a set of samples. PCA was able to obtain data representing the face at the optimum level with the data obtained from the image. The different facial and illumination levels of the same person were evaluated as the weakness point of PCA.
#
# The face recognition performend in this kernel totally based on Turk and Pentland work.
#
# Go to Contents Menu | Quick Links: 1. |3.|3.1. |3.2.|4.|4.1.|4.2.|4.3.|4.4. 4.5. |4.6. |4.7. |4.8. |4.9. |4.10.|4.11.|4.12.
#
# 3. Olivetti Dataset
#
# Brief information about Olivetti Dataset:
#
#     Face images taken between April 1992 and April 1994.
#     There are ten different image of each of 40 distinct people
#     There are 400 face images in the dataset
#     Face images were taken at different times, variying ligthing, facial express and facial detail
#     All face images have black background
#     The images are gray level
#     Size of each image is 64x64
#     Image pixel values were scaled to [0, 1] interval
#     Names of 40 people were encoded to an integer from 0 to 39
#




data=np.load("imgds_JPG.npy")
print (data.shape)
target=np.load("olivetti_faces_target.npy")

print("There are {} images in the dataset".format(len(target)))
print("There are {} unique targets in the dataset".format(len(np.unique(target))))
print("Size of each image is {}x{}".format(data.shape[1],data.shape[2],))

print("Pixel values were scaled to [0,1] interval. e.g:{}".format(data[0][0,:4]))

print("unique target number:",np.unique(target))


def show_40_distinct_people(images, unique_ids):
    # Creating 4X10 subplots in  18x9 figure size
    fig, axarr = plt.subplots(nrows=4, ncols=10, figsize=(18, 9))
    # For easy iteration flattened 4X10 subplots matrix to 40 array
    axarr = axarr.flatten()

    # iterating over user ids
    for unique_id in unique_ids:
        image_index = unique_id * 10
        axarr[unique_id].imshow(images[image_index], cmap='gray')
        axarr[unique_id].set_xticks([])
        axarr[unique_id].set_yticks([])
        axarr[unique_id].set_title("face id:{}".format(unique_id))
    plt.suptitle("There are 40 distinct people in the dataset")

show_40_distinct_people(data, np.unique(target))

#As seen in the photo gallery above, the data set has 40 different person-owned, facial images.

def show_10_faces_of_n_subject(images, subject_ids):
    cols = 10  # each subject has 10 distinct face images
    rows = (len(subject_ids) * 10) / cols  #
    rows = int(rows)

    fig, axarr = plt.subplots(nrows=rows, ncols=cols, figsize=(18, 9))
    # axarr=axarr.flatten()

    for i, subject_id in enumerate(subject_ids):
        for j in range(cols):
            image_index = subject_id * 10 + j
            axarr[i, j].imshow(images[image_index], cmap="gray")
            axarr[i, j].set_xticks([])
            axarr[i, j].set_yticks([])
            axarr[i, j].set_title("face id:{}".format(subject_id))

#You can playaround subject_ids to see other people faces
show_10_faces_of_n_subject(images=data, subject_ids=[0,8, 21, 24, 36])

#Each face of a subject has different characteristic in context of varying lighting, facial express and facial detail(glasses, beard)


#4. Machine Learning Model fo Face Recognition

#Machine learning models can work on vectors. Since the image data is in the matrix form, it must be converted to a vector.
# We reshape images for machine learnig model
X = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
print("X shape:", X.shape)


# 4.1. Split data and target into Random train and test Subsets
#
# The data set contains 10 face images for each subject. Of the face images, 70 percent will be used for training, 30 percent for testing.
# Uses stratify feature to have equal number of training and test images for each subject. Thus, there will be 7 training images and 3
# test images for each subject. You can play with training and test rates.

X_train, X_test, y_train, y_test=train_test_split(X, target, test_size=0.3, stratify=target, random_state=0)
print("X_train shape:",X_train.shape)
print("y_train shape:{}".format(y_train.shape))

y_frame=pd.DataFrame()
y_frame['subject ids']=y_train
y_frame.groupby(['subject ids']).size().plot.bar(figsize=(15,8),title="Number of Samples for Each Classes")

# 4.2.Principle Component Analysis
#
# Machine learning methods are divided into two: supervised learning and unsupervised learning. In supervised learning, the data set is divided into two main parts:
# 'data' and 'output'. The data holds the values of the sample in the data set, while the 'output' holds the class (for classification) or the target value
# (for regression). In unsupervised learning, the data set consists of only the data section.
# Non-supervised learning is generally divided into two: data transformation and clustering. In this study, the transformation of the data will be carried out
# using unsupervised learning. Unsupervised transformation methods allow for easier interpretation of data by computers and people.
# The most common unsupervised transformation applications is to reduce data size. In the size reduction process, the dimension of the data reduced.
# Princile Component Analysis (PCA) is a method that allows data to be represented in a lesser size. According to this method, the data is transformed to new components and the size of the data is reduced by selecting the most important components.
import mglearn

#mglearn.plots.plot_pca_illustration()

# The above illustration shows a simple example on a synthetic two-dimensional data set. The first drawing shows the original data points
# colored to distinguish points. The algorithm first proceeds by finding the direction of the maximum variance labeled "Component 1".
# This refers to the direction in which most of the data is associated, or in other words, the properties that are most related to each other.
#
# Then, when the algorithm is orthogonal (at right angle), it finds the direction that contains the most information in the first direction.
# There are only one possible orientation in two dimensions at a right angle, but there will be many orthogonal directions (infinite) in high dimensional spaces.

#4.3. PCA Projection of Defined Number of Target

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(X)
X_pca=pca.transform(X)

number_of_people=10
index_range=number_of_people*10
fig=plt.figure(figsize=(10,8))
ax=fig.add_subplot(1,1,1)
scatter=ax.scatter(X_pca[:index_range,0],
            X_pca[:index_range,1],
            c=target[:index_range],
            s=10,
           cmap=plt.get_cmap('jet', number_of_people)
          )

ax.set_xlabel("First Principle Component")
ax.set_ylabel("Second Principle Component")
ax.set_title("PCA projection of {} people".format(number_of_people))

fig.colorbar(scatter)
# 4.4. Finding Optimum Number of Principle Component
pca = PCA()
pca.fit(X)

plt.figure(1, figsize=(12, 8))

plt.plot(pca.explained_variance_, linewidth=2)

plt.xlabel('Components')
plt.ylabel('Explained Variaces')
#plt.show()

#In the figure above, it can be seen that 90 and more PCA components represent the same data. Now let's make the classification process using 90 PCA components.
n_components=90

pca=PCA(n_components=n_components, whiten=True)
pca.fit(X_train)

#4.5. Show Average Face
fig,ax=plt.subplots(1,1,figsize=(8,8))
ax.imshow(pca.mean_.reshape((64,64)), cmap="gray")
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Average Face')

#4.6. Show Eigen Faces
number_of_eigenfaces=len(pca.components_)
eigen_faces=pca.components_.reshape((number_of_eigenfaces, data.shape[1], data.shape[2]))

cols=10
rows=int(number_of_eigenfaces/cols)
fig, axarr=plt.subplots(nrows=rows, ncols=cols, figsize=(15,15))
axarr=axarr.flatten()
for i in range(number_of_eigenfaces):
    axarr[i].imshow(eigen_faces[i],cmap="gray")
    axarr[i].set_xticks([])
    axarr[i].set_yticks([])
    axarr[i].set_title("eigen id:{}".format(i))
plt.suptitle("All Eigen Faces".format(10*"=", 10*"="))
#Hier aufpassen


#4.7. Classification Results
X_train_pca=pca.transform(X_train)
X_test_pca=pca.transform(X_test)

clf = SVC()
clf.fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)
print("accuracy score:{:.2f}".format(metrics.accuracy_score(y_test, y_pred)))

import seaborn as sns
plt.figure(1, figsize=(12,8))
sns.heatmap(metrics.confusion_matrix(y_test, y_pred))

print(metrics.classification_report(y_test, y_pred))

#4.8. More Results

#We can get accuracy results of state of the art machine learning model.
models = []
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(("LR", LogisticRegression()))
models.append(("NB", GaussianNB()))
models.append(("KNN", KNeighborsClassifier(n_neighbors=5)))
models.append(("DT", DecisionTreeClassifier()))
models.append(("SVM", SVC()))


# According to the above results, Linear Discriminant Analysis and Logistic Regression seems to have the best performances.
for name, model in models:
    clf = model

    clf.fit(X_train_pca, y_train)

    y_pred = clf.predict(X_test_pca)
    #Einzelne augelistet
    print(10 * "=", "{} Result".format(name).upper(), 10 * "=")
    print("Accuracy score:{:0.2f}".format(metrics.accuracy_score(y_test, y_pred)))
    print()

#4.9. Validated Results

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

pca = PCA(n_components=n_components, whiten=True)
pca.fit(X)
X_pca = pca.transform(X)
for name, model in models:
    kfold = KFold(n_splits=5, shuffle=True, random_state=0)

    cv_scores = cross_val_score(model, X_pca, target, cv=kfold)
    print("{} mean cross validations score:{:.2f}".format(name, cv_scores.mean()))
#According to the cross validation scores Linear Discriminant Analysis and Logistic Regression still have best performance

lr=LinearDiscriminantAnalysis()
lr.fit(X_train_pca, y_train)
y_pred=lr.predict(X_test_pca)
#print("Accuracy score:{:.2f}".format(metrics.accuracy_score(y_test, y_pred)))

cm=metrics.confusion_matrix(y_test, y_pred)

plt.subplots(1, figsize=(12,12))
sns.heatmap(cm)

#print("Classification Results:\n{}".format(metrics.classification_report(y_test, y_pred)))

from sklearn.model_selection import LeaveOneOut
loo_cv=LeaveOneOut()
clf=LogisticRegression()
cv_scores=cross_val_score(clf,
                         X_pca,
                         target,
                         cv=loo_cv)
#print("{} Leave One Out cross-validation mean accuracy score:{:.2f}".format(clf.__class__.__name__,cv_scores.mean()))


