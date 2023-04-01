###################################################################################################################################
#GUI and MySQL
from tkinter import *          #importing library
from PIL import Image,ImageTk  #importing library 

def MAIN():
  R1=Tk()
  R1.geometry('1000x1000')        ##Setting Height&Width of the GUI form
  R1.title('Red Wine Quality Assessment') ##Setting Title of the GUI
  image=Image.open('a2.png')     ##Setting the Image 
  image=image.resize((1000,900)) ##Setting the Image Height & Width in the GUI
  photo_image=ImageTk.PhotoImage(image) ##Using Tkinter library to include Image 
  l=Label(R1, image=photo_image) ##Attaching the image inside the GUI
  l.place(x=0, y=0)              ##Initializing the place as(0,0)
  
  l=Label(R1, text="Red Wine Quality Assessment", font=('algerain',18,'bold'))
  l.place(x=300, y=40)

  b1=Button(R1, text="Linear Regression",width=18,height=2,font=('algerain',14),bg="lightblue", fg="red", command=LinearRegression)
  b1.place(x=350, y=120)

  b2=Button(R1, text="SVM",width=18,height=2,font=('algerain',14),bg="lightblue", fg="red", command=SVM)
  b2.place(x=350, y=220)

  b3=Button(R1, text="Naive Bayes",width=18,height=2,font=('algerain',14),bg="lightblue", fg="red", command=NaiveBayes)
  b3.place(x=350, y=320)

  b4=Button(R1, text="MLP",width=18,height=2,font=('algerain',14),bg="lightblue", fg="red", command=MLP)
  b4.place(x=350, y=420)

  b5=Button(R1, text="Bargraph",width=18,height=2,font=('algerain',14),bg="lightblue", fg="red", command=Bargraph)
  b5.place(x=350, y=520)

  b6=Button(R1, text="Prediction",width=18,height=2,font=('algerain',14),bg="lightblue", fg="red", command=Prediction)
  b6.place(x=350, y=620)
  
  R1.mainloop()

  
def LinearRegression():
  print('\n\n------------------LinearRegression------------------------------\n')

  import pandas as pd               ##Importing Pandas library
  import warnings                   ##Importing Warnings library
  warnings.filterwarnings('ignore') ##filtering the warnings if any

  #import dataset
  dataset = pd.read_csv('winequality-red.csv', encoding='latin1')  ## reading the CSV file data
  X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]].values             ## Setting the data from CSV file
  y = dataset.iloc[:, 11].values                                   ## Setting the data from CSV file 

  #split train and test data
  from sklearn.model_selection import train_test_split             ## Importing the train_test_split from SKLEARN
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

  #Feature Scaling
  from sklearn.preprocessing import StandardScaler                 ## Importing StandardScaler from SKLEARN
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)                              ##Applying fit_Transform to X_Train
  X_test = sc.transform(X_test)                                    ##Applying Transform to X_Test

  #apply algorithm
  from sklearn.linear_model import LinearRegression                ##Importing LinearRegression from SKLEARN
  classifier =  LinearRegression()
  classifier.fit(X_train, y_train)         

  #accuracy score
  from sklearn.metrics import accuracy_score  
  A = classifier.score(X_test,y_test)
  print('Accuracy Score: {}\n'.format(A))


def SVM():
  print('\n\n------------------SVM------------------------------\n')
  
  import pandas as pd               ##Importing Pandas library
  import warnings                   ##Importing Warnings library
  warnings.filterwarnings('ignore') ##filtering the warnings if any

  #import dataset
  dataset = pd.read_csv('winequality-red.csv', encoding='latin1')  ## reading the CSV file data
  X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]].values             ## Setting the data from CSV file 
  y = dataset.iloc[:, 11].values                                   ## Setting the data from CSV file 

  #split train and test data
  from sklearn.model_selection import train_test_split             ## Importing the train_test_split from SKLEARN
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

  #Feature Scaling
  from sklearn.preprocessing import StandardScaler                 ## Importing Standard scaler
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)                             ##Applying fit_Transform to X_Train
  X_test = sc.transform(X_test)                                   ##Applying Transform to X_Test
    
  #apply algorithm
  from sklearn.svm import SVC                                     ##Importing SVC algorithm from SKLEARN                           
  classifier =  SVC()
  classifier.fit(X_train, y_train)

  y_pred = classifier.predict(X_test) 

  #accuracy score
  from sklearn.metrics import accuracy_score                   ##Importing Accuracy Score
  A = accuracy_score(y_test,y_pred)
  print('Accuracy Score: {}\n'.format(A))


def NaiveBayes():
    print('\n\n------------------NaiveBayes------------------------------\n')

    import pandas as pd               ##Importing Pandas library
    import warnings                   ##Importing Warnings library 
    warnings.filterwarnings('ignore') ##filtering the warnings if any

    #import dataset
    dataset = pd.read_csv('winequality-red.csv', encoding='latin1')  ## reading the CSV file data
    X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]].values             ## Setting the data from CSV file
    y = dataset.iloc[:, 11].values                                   ## Setting the data from CSV file  

    #split train and test data
    from sklearn.model_selection import train_test_split             ## Importing the train_test_split from SKLEARN
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    #Feature Scaling
    from sklearn.preprocessing import StandardScaler                 ## Importing Standard scaler
    sc = StandardScaler() 
    X_train = sc.fit_transform(X_train)                              ##Applying fit_Transform to X_Train
    X_test = sc.transform(X_test)                                    ##Applying Transform to X_Test

    #apply algorithm
    from sklearn.naive_bayes import GaussianNB                       ##Importing naive_bayes algorithm from SKLEARN     
    classifier =  GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    #accuracy score
    from sklearn.metrics import accuracy_score                      ##Importing Accuracy Score 
    A = accuracy_score(y_test,y_pred)
    print('Accuracy Score: {}\n'.format(A))
    

def MLP():
  print('\n\n------------------MLP------------------------------\n')

  import pandas as pd               ##Importing Pandas library
  import warnings                   ##Importing Warnings library 
  warnings.filterwarnings('ignore') ##filtering the warnings if any

  #import dataset
  dataset = pd.read_csv('winequality-red.csv', encoding='latin1')  ## reading the CSV file data
  X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]].values             ## Setting the data from CSV file
  y = dataset.iloc[:, 11].values                                   ## Setting the data from CSV file  

  #split train and test data
  from sklearn.model_selection import train_test_split             ## Importing the train_test_split from SKLEARN
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

  #Feature Scaling
  from sklearn.preprocessing import StandardScaler                 ## Importing Standard scaler
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)                              ##Applying fit_Transform to X_Train
  X_test = sc.transform(X_test)                                    ##Applying Transform to X_Test

  #apply algorithm
  from sklearn.neural_network import MLPClassifier                 ##Importing MLP algorithm from SKLEARN   
  classifier =  MLPClassifier()
  classifier.fit(X_train, y_train)

  y_pred = classifier.predict(X_test)

  #accuracy score
  from sklearn.metrics import accuracy_score                      ##Importing Accuracy Score 
  A = accuracy_score(y_test,y_pred)
  print('Accuracy Score: {}\n'.format(A))




def Bargraph():
    print('\n------------Plotting Bargraph------------')
    import pandas as pd                ##Importing Pandas library
    import matplotlib.pyplot as plt    ##Importing matplotlib.pyplot library

    dataset = pd.read_csv('winequality-red.csv', encoding='latin1')   ## reading the CSV file data

    plt.figure(figsize=[15,6])    ##Setting the size of bargraph

    a1 = str(input('enter for x-axis:'))   ##Initializing X-axis
    a2 = str(input('enter for y-axis:'))   ##Initializing Y-axis

    plt.bar(dataset[a1],dataset[a2])       ##Applying BarGaprh on X-Axis and Y-Axis
    plt.xlabel(a1)                   
    plt.ylabel(a2)
    plt.show()



#Prediction
def Prediction(): 
  import numpy as np                 ##Importing Numpy library             
  import matplotlib.pyplot as plt    ##Importing matplotlib.pyplot library
  import pandas as pd                ##Importing Pandas library
  import warnings                     ##Importing Warnings library 
  warnings.filterwarnings('ignore')

  #import dataset
  dataset = pd.read_csv('winequality-red.csv', encoding='latin1')    ## reading the CSV file data
  X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10]].values               ## Setting the data from CSV file
  y = dataset.iloc[:, 11].values                                     ## Setting the data from CSV file

  #split train and test data
  from sklearn.model_selection import train_test_split               ## Importing the train_test_split from SKLEARN
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

  # Feature Scaling
  from sklearn.preprocessing import StandardScaler                    ## Importing Standard scaler
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)                                 ##Applying fit_Transform to X_Train
  X_test = sc.transform(X_test)                                       ##Applying Transform to X_Test

  #apply algorithm
  from sklearn.naive_bayes import GaussianNB                           
  classifier =  GaussianNB()
  classifier.fit(X_train, y_train)

  # Predicting the Test set results
  y_pred = classifier.predict(X_test)

  #quality > 6.5 => "good"
  print('\n----------Please enter the details below:-----------\n')    ##Taking the attribute values as input from user

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
  z = classifier.predict([[a,b,c,d,e,f,g,h,i,j,k]])                ##Calculating prediction usind Predict function  
  print(z)
  
  if z>=5 and z<=6.5:
    print('Red Wine Quality is Bad')                              ##specifying the Quality 
  elif z>6.5 and z<=10:
    print('Red Wine Quality is Good')                             ##specifying the Quality 
    

MAIN()
###################################################################################################################################
