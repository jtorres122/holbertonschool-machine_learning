#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))
print(fruit)

labels = ['Farrah', 'Fred', 'Felicia']
width = 0.5

apples = fruit[0][:]
bananas = fruit[1][:]
oranges = fruit[2][:]
peaches = fruit[3][:]

plt.bar(labels, apples, width, color='red')
plt.bar(labels, bananas, width,bottom=apples, color='yellow')
plt.bar(labels, oranges, width, bottom=apples+bananas, color='#ff8000')
plt.bar(labels, peaches, width, bottom=apples+bananas+oranges, color='#ffe5b4')

plt.yticks(np.arange(0, 90, step=10))

plt.ylabel("Quantity of Fruit")
plt.legend(["apples", "bananas", "oranges", "peaches"])
plt.title("Number of Fruit per Person")
plt.show()
