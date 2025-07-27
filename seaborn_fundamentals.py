import seaborn as sns
import matplotlib.pyplot as plt

""" Datasets """
# View available example datasets from seaborn
print(sns.get_dataset_names())

# Load common example datasets
tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')
titanic = sns.load_dataset('titanic')
planets = sns.load_dataset('planets')

# Scatter plot showing relationship between tip and total bill
# sns.scatterplot(x='tip', y='total_bill', data=tips, hue='day', size='size', palette='YlGnBu')

# Histogram with optional KDE overlay
# sns.histplot(tips['tip'], kde=True, bins=15)

# Bar plot of average tips by sex
# sns.barplot(x='sex', y='tip', data=tips, palette='YlGnBu')

# Box plot to show tip distributions by day and sex
# sns.boxplot(x='day', y='tip', data=tips, hue='sex', palette='YlGnBu')

# Distribution plot (jittered strip plot)
# sns.stripplot(x='day', y='tip', data=tips, hue='sex', dodge=True, palette='YlGnBu')

# Joint distribution of tip and total bill with hex bin plot
# sns.jointplot(x='tip', y='total_bill', data=tips, kind='hex', cmap='YlGnBu')

# Pairwise relationships among numeric variables
# sns.pairplot(titanic.select_dtypes(['number']), hue='pclass')

# Heatmap showing correlation matrix
# sns.heatmap(titanic.corr(), annot=True, cmap='coolwarm')

# Cluster map for hierarchical clustering (requires scipy)
# sns.clustermap(iris.drop('species', axis=1))

plt.show()
