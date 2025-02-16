# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model.functions import compute_reflection_vector

import random
qa = np.array([random.randint(-5, 5) for _ in range(3)])
qp = np.array([random.randint(-5, 5) for _ in range(3)])

qr = compute_reflection_vector(qa, qp)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.quiver(0, 0, 0, qa[0], qa[1], qa[2], color='r')
ax.quiver(0, 0, 0, qp[0], qp[1], qp[2], color='b')
ax.quiver(0, 0, 0, qr[0], qr[1], qr[2], color='y')

ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([-5, 5])

plt.show()