import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta



fileName = 'SAMSUNG.csv'

#df에 pandas.core.frame.DataFrame 타입으로 주가 정보 저장
df = pd.read_csv(fileName)

#Column 이름 설정(.csv파일과 같음)
df = df[['Open','High','Low','Close','Adj Close','Volume']]


# 0~1사이의 값으로 정규화
def MinMaxScaler(data):
    data_subtracted = data - np.min(data)
    max_minus_min = np.max(data) - np.min(data)
    return data_subtracted / (max_minus_min + 1e-7)     # 1e-7 : 0으로 나누기 방지



df_x = MinMaxScaler(df) #정규화 된 주가 정보값
df_y = df_x[['Close']]  #df_x에서 Close(종가)만 가져옴

x = df_x.values.tolist() #df_x를 리스트 형식으로
y = df_y.values.tolist()

print(df_x)







#matplotlib 라이브러리 이용해 그래프 그리기
'''
plt.figure(figsize=(14, 5)) #가로 세로 길이
plt.plot(y)                 #그래프를 그릴 데이터
plt.title('Stock Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()'''



