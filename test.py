import tensorflow as tf
from utils import *
import pandas as pd

x = [[1,2,3,4],[5,6,7,8]]
x = np.asarray(x, dtype=np.int32)
index = list(range(10))
for j in range(2):
    if True:
        np.random.shuffle(index)
    for i in range(int(2 / 2) + (1 if 2 % 2 else 0)):
        print (index[i * 2:(i + 1) * 2])

