# -*- coding:gb2312 -*-
import numpy as np
import theano.tensor as T
from theano import function

x = T.dscalar('x');
y = T.dscalar('y');
z = x + y;
f = function([x,y],z)

print f(2,3)

# 显示出z的函数

from theano import pp

print pp(z)

#做矩阵的运算

x = T.dmatrix('x') #float 64位
y = T.dmatrix('y')
z = x+y # T.dot(x,y) 乘法
f = function([x,y],z)

print f(np.arange(12).reshape((3,4)),10*np.ones((3,4)))

print '#################################'

import theano
# 激励函数

x = T.dmatrix('x')
s = 1/(1+T.exp(-x)) # np.exp() logsistic or step
logsistic = theano.function([x],s)
print logsistic([[0,1],[-2,-3]])
print '#################################'
a,b = T.dmatrices('a','b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
f = theano.function([a,b],[diff,abs_diff,diff_squared])

print f(np.ones((2,2)),np.arange(4).reshape((2, 2)))