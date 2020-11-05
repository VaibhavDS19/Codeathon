import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("music.csv")
data = np.array(data)

X = data[5:, 5:7]
y = data[5:, -1]

le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=500)

sv = SVC(kernel = 'linear').fit(X_train,y_train)

pickle.dump(sv, open('music.pkl','wb'))