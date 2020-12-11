from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

frame = {'reg': ['p', 'j', 'k', 'z'], 'total': [20, 30, 50, 60], 'light': [4, 6, 3, 5]}
df_plot = pd.DataFrame(frame)

sns.set_style('darkgrid')
sns.barplot(data=df_plot, x="reg", y='total')

plt.show()
plt.close()
