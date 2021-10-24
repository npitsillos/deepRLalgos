import timeit

setup = '''
import numpy as np
x=[np.random.rand(100) for i in range(300)]
'''

stmt = '''
data = []
for i in range(len(x)):
    data.append(x[i])
data = np.array(data)
data.mean()
'''

print(timeit.timeit(setup=setup, stmt=stmt))

setup = '''
import numpy as np
data = np.zeros((300,100))
x=[np.random.rand(100) for i in range(300)]
'''

stmt = '''
for i in range(len(x)):
    data[i] = x[i]
data.mean()
'''

print(timeit.timeit(setup=setup, stmt=stmt))
