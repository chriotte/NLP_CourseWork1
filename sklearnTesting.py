import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import style
style.use("ggplot")

x = [1,5,1.5,8,1,9]
y = [2,8,1.8,9,0.6,11]

plt.scatter(x,y)
plt.show

X =np.array([[1,2],
             [5,8],
             [1.5,1.8],
             [8,8],
             [1,0.6],
             [9,11]])

Y = [0,1,0,1,0,1]

clf = svm.SVC(kernel='linear',C = 1.0)

clf.fit(X,Y)

print("Prediction: ",clf.predict([0.58,0.76]))

