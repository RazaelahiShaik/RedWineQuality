###################################################################################################################################
#GUI and MySQL
from tkinter import *
import tkinter.messagebox
##import mysql.connector
from PIL import Image,ImageTk

def MAIN():
  R1=Tk()
  R1.geometry('800x800')
  R1.title('WELCOME-2')

  image=Image.open('a2.png')
  image=image.resize((900,700))
  photo_image=ImageTk.PhotoImage(image)
  l=Label(R1, image=photo_image)
  l.place(x=0, y=0)
  
  l=Label(R1, text="Red Wine Quality Prediction", font=('algerain',18,'bold'))
  l.place(x=220, y=40)

  b1=Button(R1, text="Linear Regression",width=18,height=2,font=('algerain',14),bg="lightblue", fg="red", command=LinearRegression)
  b1.place(x=250, y=120)

  b2=Button(R1, text="SVM",width=18,height=2,font=('algerain',14),bg="lightblue", fg="red", command=SVM)
  b2.place(x=250, y=220)

  b3=Button(R1, text="Naive Bayes",width=18,height=2,font=('algerain',14),bg="lightblue", fg="red", command=NaiveBayes)
  b3.place(x=250, y=320)

  b4=Button(R1, text="MLP",width=18,height=2,font=('algerain',14),bg="lightblue", fg="red", command=MLP)
  b4.place(x=250, y=420)

  b5=Button(R1, text="Bargraph",width=18,height=2,font=('algerain',14),bg="lightblue", fg="red", command=Bargraph)
  b5.place(x=250, y=520)

  b6=Button(R1, text="Prediction",width=18,height=2,font=('algerain',14),bg="lightblue", fg="red", command=Prediction)
  b6.place(x=250, y=620)
  
  R1.mainloop()

  
def LinearRegression():
  print('\n\n------------------LinearRegression------------------------------\n')

  import pandas as pd
  import warnings
  warnings.filterwarnings('ignore')

  #import dataset
  dataset = pd.read_csv('winequality-red.csv', encoding='latin1')
  X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]].values
  y = dataset.iloc[:, 11].values

  #split train and test data
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  #apply algorithm
  from sklearn.linear_model import LinearRegression
  classifier =  LinearRegression()
  classifier.fit(X_train, y_train)

  #accuracy score
  from sklearn.metrics import accuracy_score  
  A = classifier.score(X_test,y_test)
  print('Accuracy Score: {}\n'.format(A))


def SVM():
  print('\n\n------------------SVM------------------------------\n')
  
  import pandas as pd
  import warnings
  warnings.filterwarnings('ignore')

  #import dataset
  dataset = pd.read_csv('winequality-red.csv', encoding='latin1')
  X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]].values
  y = dataset.iloc[:, 11].values

  #split train and test data
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  #apply algorithm
  from sklearn.svm import SVC
  classifier =  SVC()
  classifier.fit(X_train, y_train)

  y_pred = classifier.predict(X_test)

  #accuracy score
  from sklearn.metrics import accuracy_score  
  A = accuracy_score(y_test,y_pred)
  print('Accuracy Score: {}\n'.format(A))


def NaiveBayes():
    print('\n\n------------------NaiveBayes------------------------------\n')

    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')

    #import dataset
    dataset = pd.read_csv('winequality-red.csv', encoding='latin1')
    X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]].values
    y = dataset.iloc[:, 11].values

    #split train and test data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #apply algorithm
    from sklearn.naive_bayes import GaussianNB
    classifier =  GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    #accuracy score
    from sklearn.metrics import accuracy_score  
    A = accuracy_score(y_test,y_pred)
    print('Accuracy Score: {}\n'.format(A))
    

def MLP():
  print('\n\n------------------MLP------------------------------\n')

  import pandas as pd
  import warnings
  warnings.filterwarnings('ignore')

  #import dataset
  dataset = pd.read_csv('winequality-red.csv', encoding='latin1')
  X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]].values
  y = dataset.iloc[:, 11].values

  #split train and test data
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  #apply algorithm
  from sklearn.neural_network import MLPClassifier
  classifier =  MLPClassifier()
  classifier.fit(X_train, y_train)

  y_pred = classifier.predict(X_test)

  #accuracy score
  from sklearn.metrics import accuracy_score  
  A = accuracy_score(y_test,y_pred)
  print('Accuracy Score: {}\n'.format(A))




def Bargraph():
    print('\n------------Plotting Bargraph------------')
    import pandas as pd
    import matplotlib.pyplot as plt

    dataset = pd.read_csv('winequality-red.csv', encoding='latin1')
    
    plt.figure(figsize=[15,6])
    plt.bar(dataset['quality'],dataset['alcohol'])
    plt.xlabel('quality')
    plt.ylabel('alcohol')
    plt.show()



#Prediction
def Prediction():
  import numpy as np
  import matplotlib.pyplot as plt
  import pandas as pd
  import warnings
  warnings.filterwarnings('ignore')

  #import dataset
  dataset = pd.read_csv('winequality-red.csv', encoding='latin1')
  X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]].values
  y = dataset.iloc[:, 11].values

  #split train and test data
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
  #print(X_train)
  #print(X_test)
  #print(y_train)
  #print(y_test)

  # Feature Scaling
  from sklearn.preprocessing import StandardScaler
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  #apply algorithm
  from sklearn.naive_bayes import GaussianNB
  classifier =  GaussianNB()
  classifier.fit(X_train, y_train)

  # Predicting the Test set results
  y_pred = classifier.predict(X_test)

  #quality > 6.5 => "good"
  print('\n----------Please enter the details below:-----------\n')

  a = float(input('\nfixed acidity:'))
  b = float(input('volatile acidity:'))
  c = float(input('citric acid:'))
  d = float(input('residual sugar:'))
  e = float(input('chlorides:'))
  f = int(input('free sulfur dioxide:'))
  g = int(input('total sulfur dioxide:'))
  h = float(input('density:'))
  i = float(input('pH:'))
  j = float(input('sulphates:'))
  k = float(input('alcohol:'))

  print('\n------------Prediction started------------')
  z = classifier.predict([[a,b,c,d,e,f,g,h,i,j,k]])
  if z>=5 and z<=6.5:
    print('Red Wine Quality is Bad')
  elif z>6.5 and z<=10:
    print('Red Wine Quality is Good')
    

MAIN()
###################################################################################################################################
