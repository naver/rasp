import pandas as pd
import seaborn as sns
sns.set_theme(style="whitegrid")
import numpy as np

import matplotlib.pyplot as plt

# color blind friendly palette
#cb_colors = [
#		'#332288',
#		'#117733',
#		'#882255',
#		'#88CCEE',
#		'#DDCC77',
#		'#CC6677',
#		'#AA4499',
#		'#44AA99']
#sns.set_palette(sns.color_palette(cb_colors)) # not working, not sure why


data_ = pd.read_csv('./wsci-sis-ablations-gains.csv')

print(data_)
data_ =  data_[151:191].iloc[:,:5]
print(data_)

data_.rename(columns={
		'\\tau ablation':'class_name', 'plot':'gain', 'Unnamed: 2':'dataset',
		'Unnamed: 3':'benchmark', 'Unnamed: 4':'setting'}, inplace=True )

data_ = data_.astype({
		'class_name':'str','gain':'float', 'dataset': 'str',
		'benchmark':'str', 'setting': 'str'})

#data_ = data_.loc[(data_['benchmark']=='10_2') & (data_['setting']=='disjoint')]
data_ = data_.loc[(data_['benchmark']=='10_5') & (data_['setting']=='disjoint')]
#data_ = data_.loc[(data_['benchmark']=='10_2') & (data_['setting']=='overlap')]
#data_ = data_.loc[(data_['benchmark']=='10_5') & (data_['setting']=='overlap')]

data_ = data_.sort_values(by=['gain'], ascending=False)


fig, ax = plt.subplots(1,1)

# green --> #73C6B6
# red --> #F1948A

#colors = ['#53AFD5' if c >= 0 else '#CC6677' for c in data_['gain'].unique()] # blue and red
colors = ['#73C6B6' if c >= 0 else '#F1948A' for c in data_['gain'].unique()] # green and red

sns.barplot(
		data=data_,# markers=True, dashes=True,
		x="gain", y="class_name", ax=ax, palette=colors,
		linewidth=1, edgecolor="0")#,
		#color='#332288', 
		#legend=None)

for container in ax.containers:
	ax.bar_label(container)

# ours solid, wilson dashed
# nice palette, possibly color blind friendly
# add final miou values somewhere

#for i_ in np.sort(data_.task_id.unique())[1:-1]:
#for i_ in np.sort(data_.task_id.unique()):
#	plt.vlines(i_, y_min, y_max, linestyles ="dashed", colors ="k", linewidth=0.3)

#legend = plt.legend(loc="lower left", edgecolor="black")#, ncol=5)
#legend.get_frame().set_alpha(None)
#legend.get_frame().set_facecolor((1, 1, 1, 1))

x_min = -15.0
x_max = 15.0
plt.xlim([x_min, x_max])

plt.show()


