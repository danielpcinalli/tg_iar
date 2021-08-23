import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator

results : pd.DataFrame = pd.read_csv('log_results.csv', header=None)

# results['max'] = results.max(axis=1)
results['mean'] = results.mean(axis=1)

sns.lineplot(data=results[['mean']])
plt.title('Fitness médio por geração')
plt.ylabel('Fitness')
plt.xlabel('Geração')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
leg=plt.legend()
leg.get_texts()[0].set_text('Fitness médio')
# plt.show()
plt.savefig('plot.pdf')
