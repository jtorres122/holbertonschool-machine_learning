#!/usr/bin/env python3
'''Module prints a simple a line graph'''
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3
x = np.arange(0, 11)

plt.plot(x, y)
plt.show()
