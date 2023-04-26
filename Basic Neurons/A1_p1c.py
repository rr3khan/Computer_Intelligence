# # ECE 457B Computer intelligence Assignment 1 Riyad Khan
# # Q1 C

import numpy as np
import matplotlib.pyplot as plt

# import cm for cool colourations for our plot
from matplotlib import cm

# Create a 3D figure
fig = plt.figure()
fig.suptitle("Adaline Plane and Data Plot", fontsize=16)
ax = fig.add_subplot(111, projection="3d")
fig2 = plt.figure(2)
fig2.suptitle("Perceptron Plane and Data Plot", fontsize=16)
ax_perceptron = fig2.add_subplot(111, projection="3d")

# Class C1=0
x1 = [0.8, 0.7, 1.2]
x2 = [-0.8, -0.7, 0.2]
x3 = [-0.5, 0.3, -0.2]
x4 = [-2.8, -0.1, -2]

c1_x = [x1[0], x2[0], x3[0], x4[0]]
c1_y = [x1[1], x2[1], x3[1], x4[1]]
c1_z = [x1[2], x2[2], x3[2], x4[2]]
# ax.scatter(x1, x2, x3, x4, c='r', marker='o')
ax.scatter(c1_x, c1_y, c1_z, c="r", marker="o", label="Class 0")
ax_perceptron.scatter(c1_x, c1_y, c1_z, c="r", marker="o", label="Class 0")

# Class C2=1
y1 = [1.2, -1.7, 2.2]
y2 = [-0.8, -2, 0.5]
y3 = [-0.5, -2.7, -1.2]
y4 = [2.8, -1.4, 2.1]

c2_x = [y1[0], y2[0], y3[0], y4[0]]
c2_y = [y1[1], y2[1], y3[1], y4[1]]
c2_z = [y1[2], y2[2], y3[2], y4[2]]
ax.scatter(c2_x, c2_y, c2_z, c="b", marker="^", label="Class 1")
ax_perceptron.scatter(c2_x, c2_y, c2_z, c="b", marker="^", label="Class 1")
ax.legend()
ax_perceptron.legend()

# Define the equation of the plane for the adaline
a = 3.93736347
b = -6.43146185
c = 0.32111855
d = 5.50878791

# Define the equation of the plane for the perceptron
ap = 0.4022965
bp = -1.42116563
cp = 0.39182569
dp = 1.50476883

# 1.17630499  0.92042525 -2.07708296  0.41901031

# Create a meshgrid of points in the x-y plane
x, y = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))

# Compute the corresponding z-coordinates
# z = (-d - a*x - b*y) / c
# z = (d - a*x - b*y) / c

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)

ax_perceptron.set_xlabel("X")
ax_perceptron.set_ylabel("Y")
ax_perceptron.set_zlabel("Z")
ax_perceptron.title
ax_perceptron.set_xlim(-5, 5)
ax_perceptron.set_ylim(-5, 5)
ax_perceptron.set_zlim(-5, 5)

# Create a mesh grid
X, Y = np.meshgrid(x, y)
Z = (d - a * X - b * Y) / c

# plane equation for perceptron
Zp = (dp - ap * X - bp * Y) / cp

# Plot the data points with scatter
# ax.scatter(x, y, z, c='b', marker='o')

# Plot the plane using plot_surface
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
ax_perceptron.plot_surface(X, Y, Zp, cmap=cm.coolwarm)
plt.legend()
plt.show()
