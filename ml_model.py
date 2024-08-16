from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


import pandas as pd
from sklearn.model_selection import train_test_split
import pickle 


df = pd.read_csv("diabetes.csv")

x = df.drop(["Outcome"],axis=1)
y = df["Outcome"]


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)


knn = KNeighborsClassifier(n_neighbors=12)
LR = LogisticRegression()



knn.fit(x_train,y_train)
LR.fit(x_train,y_train)


print(f"{knn.score(x_test,y_test)}")
print(f"{LR.score(x_test,y_test)}")


pickle.dump(knn,open("knn_model.pkl","wb"))
pickle.dump(LR,open("LR_model.pkl","wb"))