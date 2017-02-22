# -*- coding:gb2312 -*-
import numpy as np
import theano.tensor as T
import theano

state = theano.shared(np.array(0, dtype=np.float64), 'state')
inc = T.scalar('inc', dtype=state.dtype)
accunmulator = theano.function([inc], state, updates=[(state, state + inc)])

print (accunmulator(10))
print (accunmulator(1))
print (accunmulator(1))

# get方法
print (state.get_value())
accunmulator(1)
print (state.get_value())
accunmulator(10)
print (state.get_value())
# set方法
state.set_value(-1)
accunmulator(10)
print (state.get_value())

tmp_func = state * 2 + inc
a = T.scalar(dtype=state.dtype)
skip_shared = theano.function([inc, a], tmp_func, givens=[(state, a)])
