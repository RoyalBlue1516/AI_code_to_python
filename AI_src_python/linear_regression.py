import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

diabetes = datasets.load_diabetes()

x_data=diabetes.data
y_data=diabetes.target

x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.15, random_state=42)
x_dev, x_test, y_dev, y_test = train_test_split(x_temp, y_temp, test_size=0.67, random_state=42)

class MultiLinear:
  def __init__(self,learning_rate=0.001):
    self.w=None # weight
    self.b=None # bias
    self.lr=learning_rate #모델의 학습률
    self.best_loss=float("inf")#
    self.early_stopping=0

  def forward(self,x):
    y_pred=np.sum(x*self.w)+self.b #np.sum함수는 인자로 받은 numpy배열의 모든 원소의 합을 return
    return y_pred

  def loss(self,x,y):
    y_pred=self.forward(x)
    return (y_pred-y)**2

  def gradient(self,x,y):
    y_pred=self.forward(x)
    w_grad=2*x*(y_pred-y)
    b_grad=2*(y_pred-y)

    return w_grad,b_grad

  def val_loss(self,x_dev,y_dev):
    val_loss=0
    for x,y in zip(x_dev,y_dev):
      val_loss+=self.loss(x,y)
    return val_loss/len(y_dev)

  def fit(self,x_data,y_data,epochs=100, x_dev=None, y_dev=None,x_test=None,y_test=None, minibatch_size=50):
    self.w=np.ones(x_data.shape[1]) #weight들을 전부 1로 초기화
    self.b=0 # bias를 0으로 초기화
    for epoch in range(epochs):
      l=0 #계산할 손실값
      w_grad=np.zeros(x_data.shape[1]) #weight의 기울기를 누적할 numpy배열
      b_grad=0  #bias의 기울기를 누적할 변수

      for i in range(0, len(y_data) ,minibatch_size):
        x_batch = x_data[i:i + minibatch_size]
        y_batch = y_data[i:i + minibatch_size]        

        for x,y in zip(x_batch,y_batch):
          l+=self.loss(x,y)
          w_i,b_i=self.gradient(x,y)

          w_grad+=w_i #weight누적
          b_grad+=b_i #bias누적

        self.w-=self.lr*(w_grad/len(y_batch)) #weight 업데이트
        self.b-=self.lr*(b_grad/len(y_batch)) #bias 업데이트

      val_loss=self.val_loss(x_dev,y_dev)
      MSE_test=self.val_loss(x_test,y_test)

      #early stopping
      if val_loss<self.best_loss:
        self.best_loss=val_loss
      else:
        self.early_stopping+=1
        
      if self.early_stopping==5:
        print(f'best_val_loss: {self.best_loss:.4f}')
        print('early-stop')
        break
      
      print(f'epoch ({epoch+1}) ===> MSE_train : {l/len(y_data):.4f} | w : {self.w[0]:.4f} | b : {self.b:.4f} | MSE_dev : {val_loss:.4f} | MSE_test : {MSE_test:.4f}')

      

model=MultiLinear(learning_rate=0.1)
model.fit(x_train,y_train,epochs=100,x_dev=x_dev,y_dev=y_dev,x_test=x_test,y_test=y_test)