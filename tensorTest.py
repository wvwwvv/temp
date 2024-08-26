import tensorflow as tf
import numpy as np
#경고 메시지 지우기
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

hello = tf.constant([[1,2,3,4,5],[1,3,5,7,9]])
print('-----------------------------')

hi = hello[:,0:2]
print(hi)
