# Machine-learning

pla.py:
进入终端
import pla
>>> data = pla.makeLinearSeparableData([4,3],100)
>>> w = pla.train(data)
>>> w
array([[ 16.32172416,  11.54429628]])
>>> w = pla.train(data, True)
