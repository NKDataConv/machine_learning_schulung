import numpy as np

# 1. Schnell
# 2. Speichereffizient
# 3. Vektorwertig

my_list = [1, 2, 3]
my_list = [element +1 for element in my_list]
my_list

array = np.array([1, 2, 3])
array + 1
array * 2

array.shape

array_two_dimensional = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
array_two_dimensional.shape
array_two_dimensional.dtype

array_two_dimensional.transpose()
array_two_dimensional.transpose().shape

array.mean()
array.max()
array.std()
array.min()

array_two_dimensional.mean(axis=0)

array_two_dimensional.reshape((3, 2))

