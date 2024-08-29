import numpy as np
from numpy import array
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM


#경고 메시지 지우기
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = array([4,5,6,7])

print('x shape ',x.shape)# (4,3)
print('y shape : ',y.shape)# (4,)
#  x  y
# 123 4
# 234 5
# 345 6
# 456 7


print('-------x reshape-----------') #LSTM의 입력으로는 3차원 데이터가 필요함
x=x.reshape((x.shape[0],x.shape[1],1))# (4,3,1) reshape 전체 곱 수 같아야 4*3=4*3*1
#  x        y
# [1][2][3] 4
# .....


# 2. 모델 구성
model =Sequential()
model.add(LSTM(10,activation='relu',input_shape=(3,1)))
# DENSE와 사용법 동일하나 input_shape=(열(학습시키는 데이터인 x의 열의 개수), 몇개씩잘라작업(이는 x.shape[2]와 동일))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

# 3. 실행
model.compile(optimizer='adam',loss='mse') #최적화 함수와 손실함수
model.fit(x,y,epochs=100,batch_size=1) #100번 학습, batch_size는 한 번에 처리하는 데이터 샘플의 수

x_input= np.array([[6,7,8],[7,8,9]])
x_input=x_input.reshape((2,3,1))
print(x_input.shape)

yhat=model.predict(x_input)
print(yhat)