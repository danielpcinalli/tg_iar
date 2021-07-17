import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

results : pd.DataFrame = pd.read_csv('log_results.csv', header=None)

results['max'] = results.max(axis=0)
results['mean'] = results.mean(axis=0)

sns.lineplot(data=results[['max', 'mean']])
plt.show()
print(results)
