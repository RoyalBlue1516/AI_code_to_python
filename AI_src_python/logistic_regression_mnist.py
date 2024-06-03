import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split 

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_train,x_dev,y_train,y_dev=train_test_split(x_train,y_train,stratify=y_train,test_size=0.15)


x_train = x_train.reshape(x_train.shape[0],28*28) 
x_dev=x_dev.reshape(x_dev.shape[0],28*28) 
x_test = x_test.reshape(x_test.shape[0],28*28) 

# 0~255 값 분포를 0~1 사이에 분포하도록 바꿈 
x_train = x_train.astype('float32') / 255. 
x_dev=x_dev.astype('float32')/255. 
x_test = x_test.astype('float32') / 255. 

# one-hot encoding 
y_train = keras.utils.to_categorical(y_train, 10) 
y_dev=keras.utils.to_categorical(y_dev,10) 
y_test = keras.utils.to_categorical(y_test, 10)

class Multiclass:
  def __init__(self,learning_rate=0.01):
    self.w=None #가중치 행렬
    self.b=None #바이어스 배열
    self.lr=learning_rate #학습률 
    self.best_loss=float("inf")#검증 손실값과 비교값
    self.early_stopping=0 #early_stopping 횟수값

  def forward(self,x):
    z=np.dot(x,self.w)+self.b
    z=np.clip(z,-100,None) #NaN 방지
    return z

  def softmax(self,z):
    exp_z=np.exp(z)
    a=exp_z/np.sum(exp_z)
    return a

  def loss(self,x,y):
    z=self.forward(x)
    a=self.softmax(z)
    return -np.sum(y*np.log(a)) #손실 계산 후 리턴

  def gradient(self,x,y):
    z=self.forward(x) #선형방정식을 통한 z값 산출
    a=self.softmax(z) #z값을 softmax에 통과시켜 a값 산출

    w_grad=-np.dot(x.reshape(-1,1),(y-a).reshape(1,-1)) #가중치의 기울기
    b_grad=-(y-a) #바이어스의 기울기

    return w_grad,b_grad

  def fit(self,x_data,y_data,epochs=100,x_dev=None,y_dev=None,x_test=None,y_test=None, minibatch_size=100):
    
    for epoch in range(epochs):
      l=0 #손실값을 계산할 변수
      for i in range(0, len(y_data) ,minibatch_size):
        x_batch=x_data[i:i+minibatch_size,:] 
        y_batch=y_data[i:i+minibatch_size,:]
        self.w=np.random.normal(0,1,(x_batch.shape[1],y_batch.shape[1])) #표준정규분포로 초기화 
        self.b=np.zeros(y_batch.shape[1]) #0으로 초기화
        w_grad=np.zeros((x_batch.shape[1],y_batch.shape[1])) #손실함수에 대한 가중치의 기울기
        b_grad=0 #손실함수에 대한 바이어스의 기울기
        
        for x,y in zip(x_batch,y_batch):
          l+=self.loss(x,y) #매 에포크마다 손실값 계산

          w_i,b_i=self.gradient(x,y) #가중치와 바이어스의 기울기를 계산

          w_grad+=w_i #가중치의 기울기 누적
          b_grad+=b_i #바이어스의 기울기 누적 

        self.w-=self.lr*(w_grad/len(y_batch)) #가중치 업데이트
        self.b-=self.lr*(b_grad/len(y_batch)) #바이어스 업데이트

      val_loss=self.val_loss(x_dev,y_dev)
      train_accuracy = self.score(x_data,y_data)
      val_accuracy = self.score(x_dev,y_dev)
      test_accuracy = self.score(x_test,y_test)


      if val_loss<self.best_loss:
        self.best_loss=val_loss
      else:
        self.early_stopping+=1
        
      if self.early_stopping==5:
        print(f'best_val_loss: {self.best_loss:.4f}')
        print('early-stop')
        break
      
      print(f'epoch({epoch+1}) ===> loss : {l/len(y_data):.4f} | val_loss : {val_loss:.4f} | train_accuracy : {train_accuracy:.4f} | dev_accuracy : {val_accuracy:.4f} | test_accuracy : {test_accuracy:.4f}')


  def predict(self,x_data):
    z=self.forward(x_data)
    return np.argmax(z,axis=1) #가장 큰 인덱스 리턴

  def score(self,x_data,y_data):
    return np.mean(self.predict(x_data)==np.argmax(y_data,axis=1))

  def val_loss(self,x_dev,y_dev):
    val_loss=0
    for x,y in zip(x_dev,y_dev):
      val_loss+=self.loss(x,y)

    return val_loss/len(y_dev)


model=Multiclass(learning_rate=1.0) 
model.fit(x_train,y_train,epochs=100,x_dev=x_dev,y_dev=y_dev,x_test=x_test,y_test=y_test)