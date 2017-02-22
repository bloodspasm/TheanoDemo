# -*- coding:gb2312 -*-
import numpy as np
import theano.tensor as T
from theano import function

x = T.dscalar('x');  # ���� T.scalar(dtype=float64)
y = T.dscalar('y');
z = x + y;
f = function([x, y], z)

print f(2, 3)

# ��ʾ��z�ĺ���

from theano import pp

print pp(z)

# �����������

x = T.dmatrix('x')  # ���� float 64λ
y = T.dmatrix('y')
z = x + y  # T.dot(x,y) �˷�
f = function([x, y], z)

print f(np.arange(12).reshape((3, 4)), 10 * np.ones((3, 4)))

print '#################################'

import theano

# ��������

x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))  # np.exp() logsistic or step
logsistic = theano.function([x], s)
print (logsistic([[0, 1], [-2, -3]]))
print ('#################################')
a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff ** 2
f = theano.function([a, b], [diff, abs_diff, diff_squared])

print (f(np.ones((2, 2)), np.arange(4).reshape((2, 2))))
print ('#################################')

x, y, w = T.dscalars('x', 'y', 'w')
z = (x + y) * w
f = theano.function([x,
                     theano.In(y, value=1),
                     theano.In(w, value=2, name='weights')],
                    z)
print(f(23, 2, weights=10))
# #�ȼ���
# def f(a,b = 1,c =2)
#     return (a+b)*c
