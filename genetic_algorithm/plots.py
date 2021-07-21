import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

results : pd.DataFrame = pd.read_csv('log_results.csv', header=None)

results['max'] = results.max(axis=1)
results['mean'] = results.mean(axis=1)

sns.lineplot(data=results[['max', 'mean']])
print(results)
plt.show()
