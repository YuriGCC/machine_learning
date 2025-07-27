# NumPy version 2.2.4
import numpy as np
import random

""" Arrays """
a = np.array([[[1], [2], [3]], [[4], [5], [6]]])

ab = [[1, 2, 3], [4, 5, 6]]

print(ab[0][2])   # Accessing raw list
print(a[0, 2])    # Accessing NumPy array

# Shape of the array: tuple with dimensions (rows, columns, ...)
print(a.shape)  # Output: (2, 3, 1) → 2 blocks, 3 rows, 1 column

# Number of dimensions (a.k.a. rank of the array)
print(a.ndim)   # Output: 3 → 3D array

# Total number of elements (product of dimensions)
print(a.size)   # Output: 6

# Data types in NumPy arrays
datas = np.array(['1', 1, 1.001, False])  # Mixed types become a single dtype
datas2 = np.array([10.1, 10.2, 11.3, 1.1], dtype=np.float64)

print(datas2.dtype)  # float64
print(type(datas[0]))  # numpy.str_
print(datas[0].dtype)  # <U32 or similar depending on content

""" NumPy Data Types """

a = np.full((3, 3), random.randint(0, 3))  # 3x3 matrix filled with a random int between 0–3
zeroarray = np.zeros((2, 3))   # Array filled with 0s
onearray = np.ones((2, 3))     # Array filled with 1s
empty = np.empty((2, 3))       # Creates array without initializing values (may contain garbage)

print(zeroarray)
print(onearray)
print(empty)

x_values = np.arange(0, 100, 5)
x_values2 = [float(e**2) for e in x_values]
print(x_values)
print(x_values2)

y_values = np.linspace(0, 100, 4)  # Evenly spaced values between 0 and 100 (4 points)
print(y_values)

""" NaN and Infinity """

print(np.nan)
print(np.inf)

print(np.isnan(np.nan))  # True
print(np.isnan(np.inf))  # False

# These produce warnings or errors:
# print(np.sqrt(-1))
# print(np.array([10]) / 0)

""" Math operations
    For operations between arrays, they must have the same shape
"""
l1 = [1, 2, 3, 4, 5]
l2 = [6, 7, 8, 9, 0]

a1 = np.array(l1)
a2 = np.array(l2)

print(l1 * 5)   # List repetition
print(a1 * 5)   # Element-wise multiplication

a3 = [[1, 2, 3], [4, 5, 6]]
a4 = [[1, 2, 3], [4, 5, 6]]
l3 = np.array([[1, 2, 3], [4, 5, 6]])
l4 = np.array([[1, 2, 3], [4, 5, 6]])

print(a3 + a4)     # List concatenation
print(l3 + l4)     # Element-wise addition

row_vector = np.array([1, 2, 3])
column_vector = np.array([[1], [2]])
print(column_vector + row_vector)  # Broadcasting

a = np.array([[1, 2, 3], [4, 5, 6]])
print(np.sqrt(a))  # Square root
print(np.cos(a))   # Cosine

""" Array methods """

a = np.array([1, 2, 3])
print(np.append(a, [7, 8, 9]))  # Returns a new array
print(a)  # Original array remains unchanged
a = np.append(a, [7, 8, 9])     # Now it's updated

print(np.insert(a, 3, [4, 5, 6]))  # Insert at position 3
print(np.delete(a, 1))            # Delete element at position 1

b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(b)
print(np.delete(b, 5))        # Flattened delete (5th element)
print(np.delete(b, 0, axis=0))  # Delete first row

""" Structural methods (change array shape) """

matrix = np.array([[1, 2, 3, 4], [4, 5, 6, 4], [7, 8, 9, 4]])

print(matrix.shape)
print(matrix.reshape(4, 3).shape)
print(matrix.reshape(matrix.size,))        # Flatten
print(matrix.reshape(1, matrix.size))      # Row vector
print(matrix.reshape(matrix.size, 1))      # Column vector
print(matrix.reshape(3, 1, 4))             # 3D reshape
print(matrix.reshape(3, 1, 4).shape)

# Only valid if total size matches (e.g., 1 * 3 * 2 * 2 = 12)
# print(matrix.reshape(1, 3, 2, 2))

# Resize modifies the array in-place
np.resize(matrix, (6, 2))
print(matrix)
np.resize(matrix, (3, 4))  # No effect unless reassigned

# Flattening arrays
print(matrix.flatten())  # Returns a copy
print(matrix.ravel())    # Returns a view (whenever possible)

# Difference: flatten() doesn't affect the original
var1 = matrix.flatten()
var1[0] = 100
print(var1)
print(matrix)

# ravel() modifies original if possible
var1 = matrix.ravel()
var1[0] = 100
print(var1)
print(matrix)

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
flat_list = [int(v) for v in matrix.flat]
print(flat_list)

matrix2 = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix2.transpose())  # Transpose 2D
print(matrix2.T)
print(matrix2.swapaxes(0, 1))  # Equivalent to transpose for 2D

""" Concatenating and splitting arrays """

array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
# Concatenate (axis=0 by default)
c = np.concatenate((array1, array2), axis=0)
print(c)

# Stack creates a new axis (like stacking on top)
c = np.stack((array1, array2))
print(c)

# Horizontal stack
c = np.hstack((array1, array2))

# Splitting arrays
two_d_array = np.array([[1, 2, 3], [4, 5, 6]])
print(np.split(two_d_array, 2, axis=0))

""" Aggregation functions """

vx = np.array([
    [10, 20, 30, 40, 50],
    [15, 25, 35, 45, 55],
    [11, 22, 33, 44, 55],
    [12, 24, 36, 48, 60],
    [13, 26, 39, 52, 65]
])

print(vx.min())
print(vx.max())
print(vx.mean())
print(vx.sum())
print(np.median(vx))

""" NumPy random """

number = np.random.randint(100)
print(number)

# Range: [0, 10), shape: 2x3x4
numbers = np.random.randint(0, 10, size=(2, 3, 4))
print(numbers)

# Normally distributed random values (mean=170, std=15)
heights = np.random.normal(loc=170, scale=15, size=(5, 10))
print(heights)

# Choose random values from a set
choices = np.random.choice([10, 20, 30, 40, 50], size=(5, 10))
print(choices)

""" Saving and loading arrays """

# Export array to .npy
# exported_array = np.random.choice([10, 20, 30, 40, 50], size=(5, 10))
# np.save("myarray.npy", exported_array)

# Load from .npy
# imported_array = np.load("myarray.npy")
# print(imported_array)

# Save to CSV
# np.savetxt("array.csv", exported_array, delimiter=",")

# Load from CSV
# imported_array = np.loadtxt("array.csv", delimiter=",")
# print(imported_array)
