import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from matplotlib.lines import lineStyles  # (Not used in this script)
import random
from IPython.display import display, clear_output

""" Basic charts """

# Scatter plot
x_data = np.random.uniform(-100, 101, size=50)
y_data = np.random.uniform(-100, 101, size=50)

# plt.scatter(x_data, y_data, c='black', marker='*', s=150, alpha=0.3)

# Line plot
years = [2006 + x for x in range(16)]
weights = [80, 83, 84, 85, 86, 82, 81, 79, 83, 80, 82, 82, 82, 81, 80, 79]
# plt.plot(years, weights, c='red', lw=3, linestyle='--')

# Bar chart
x = ['C++', 'C#', 'Python', 'Java', 'Go']
y = [20, 50, 140, 1, 42]
# plt.bar(x, y, color='g', align='edge', width=0.5, edgecolor='gray', lw=6)

# Histogram
ages = np.random.normal(20, 1.5, 100)
# plt.hist(ages, bins=[ages.min(), 18, 21, ages.max()], cumulative=True)

# Pie chart
languages = ['C++', 'C#', 'Python', 'Java', 'Go']
votes = [50, 24, 14, 6, 17]
explodes = [0.3, 0, 0, 0, 0]
# plt.pie(votes, labels=languages, explode=explodes, autopct="%.2f%%", pctdistance=1.5, startangle=90)

# Box plot
heights = np.random.normal(172, 8, 300)
# plt.boxplot(heights)

# Box plot for concatenated data ranges
first = np.linspace(0, 10, 25)
second = np.linspace(10, 200, 25)
third = np.linspace(200, 210, 25)
fourth = np.linspace(210, 230, 25)

data = np.concatenate((first, second, third, fourth))
# plt.boxplot(data)

""" Chart customization """

years = [2014 + x for x in range(8)]
income = [55, 56, 62, 61, 72, 72, 73, 75]

income_ticks = list(range(50, 81, 2))

# Title, X and Y labels
# plt.title("Income in dollars", fontsize=30, fontname="FreeSerif")
# plt.xlabel("Years")
# plt.ylabel("Yearly Income in USD")

# Line plot with customized ticks
# plt.plot(years, income)
# plt.yticks(income_ticks, [f"{x}k USD" for x in income_ticks])

# Multiple lines in the same plot
stock_a = np.random.randint(0, 75, 50)
stock_b = np.random.randint(0, 50, 50)
stock_c = np.random.randint(0, 25, 50)

# plt.plot(stock_a, label='Branch 1 Stock')
# plt.plot(stock_b, label='Branch 2 Stock')
# plt.plot(stock_c, label='Branch 3 Stock')
# plt.legend(loc='lower center')

votes = [10, 2, 5, 16, 22]
people = ["A", "B", "C", "D", "E"]

# plt.pie(votes, labels=None)
# style.use("ggplot")
# plt.legend(labels=people)

""" Multiple plots """

# Different charts in separate windows
x1, y1 = np.random.random(100), np.random.random(100)
x2, y2 = np.arange(100), np.random.random(100)

# plt.figure(1)
# plt.scatter(x1, y1)
# plt.figure(2)
# plt.plot(x2, y2)

# Multiple plots in the same window using subplots
x = np.arange(100)

# fig, axis = plt.subplots(2, 2)

# axis[0, 0].plot(x, np.sin(x))
# axis[0, 0].set_title("Sine Wave")

# axis[0, 1].plot(x, np.cos(x))
# axis[0, 1].set_title("Cosine Wave")

# axis[1, 0].plot(x, np.random.random(100))
# axis[1, 0].set_title("Random Function")

# axis[1, 1].plot(x, np.log(x))
# axis[1, 1].set_title("Log Function")

""" Exporting charts """

# Save chart to image file
# plt.savefig("four_charts.png", dpi=300, transparent=True)
# plt.tight_layout()
# plt.show()

""" 3D Charts """

# ax = plt.axes(projection="3d")

x = np.arange(0, 50, 0.1)
y = np.sin(x)
z = np.cos(x)

# 3D line chart
# ax.plot(x, y, z)
# ax.set_title("3D plot")
# ax.set_xlabel("X")
# ax.set_ylabel("COS(X)")
# ax.set_zlabel("SIN(X)")

# 3D surface chart
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# ax.plot_surface(X, Y, Z, cmap="Spectral")
# ax.set_title('Surface plot')
# ax.view_init(azim=0, elev=90)  # top-down view

""" Animating charts """

# Simulating coin flips animation
heads_tails = [0, 0]

for _ in range(1000):
    heads_tails[random.randint(0, 1)] += 1
    plt.bar(["Heads", "Tails"], heads_tails, color=["red", "blue"])
    # Animation using pause inside the loop
    plt.pause(0.001)

plt.show()
