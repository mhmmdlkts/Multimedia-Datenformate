from sklearn.datasets import fetch_olivetti_faces
  
faces = fetch_olivetti_faces(shuffle=True, random_state=1000)
X_train = faces['images']
