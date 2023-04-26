# ECE 457B Computer intelligence Assignment 1 Riyad Khan
# Q2

import numpy as np
import matplotlib.pyplot as plt

# import cm for cool colourations for our plot
from matplotlib import cm


# Create a 3D figure
fig = plt.figure()
fig.suptitle("XNOR Boundary Plot", fontsize=16)
ax = fig.add_subplot(111)
# ax2 = fig.add_subplot(111)

# Class -1
x1 = [0, 0]
x2 = [1, 1]

c1_x = [x1[0], x2[0]]
c1_y = [x1[1], x2[1]]

# ax.scatter(x1, x2, x3, x4, c='r', marker='o')
ax.scatter(c1_x, c1_y, c="b", marker="^", label="Class -1")
# ax2.scatter(c1_x, c1_y, c='r', marker='o', label="Class 0")

# Class 1
y1 = [1, 0]
y2 = [0, 1]


c2_x = [y1[0], y2[0]]
c2_y = [y1[1], y2[1]]
# c2_z = [y1[2], y2[2]]
ax.scatter(c2_x, c2_y, c="r", marker="o", label="Class 1")
# ax2.scatter(c2_x, c2_y, c2_z, c='b', marker='^', label="Class 1")
ax.legend()


# 1.17630499  0.92042525 -2.07708296  0.41901031


# Compute the corresponding z-coordinates
# z = (-d - a*x - b*y) / c
# z = (d - a*x - b*y) / c

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 2)

# Plot the boundary lines

a1 = np.linspace(-2, 10, 1000)
plt.plot(-a1, a1 + 0.1, linestyle="solid", label="A1")

a2 = np.linspace(-2, 10, 1000)
plt.plot(-a2, a2 + 1.5, linestyle="solid", label="A2")

plt.grid()
plt.legend()
plt.show()
