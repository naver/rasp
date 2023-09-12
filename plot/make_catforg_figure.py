import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

# color blind friendly palette
cb_colors = [
		'#332288',
		'#117733',
		'#882255',
		'#88CCEE',
		'#DDCC77',
		'#CC6677',
		'#AA4499',
		'#44AA99']
#sns.set_palette(sns.color_palette(cb_colors)) # not working, not sure why


data_ = pd.read_csv('./wsci-sis-results.csv')

data_ =  data_[65:].iloc[:,:7]

data_.rename(columns={
		'Unnamed: 0':'method', 'Unnamed: 1':'step', 'Unnamed: 2':'task_id',
		'Unnamed: 3':'miou', 'Unnamed: 4':'dataset', 'Unnamed: 5':'benchmark',
		'Unnamed: 6':'setting'}, inplace=True )

print(data_)

data_ = data_.astype({
		'method':'str','step':'int', 'task_id':'int',
		'miou':'float', 'dataset':'str', 'benchmark':'str',
		'setting':'str'})

# only keep one method
#data_ = data_[data_.method=='Ours']

#data_ = data_.loc[(data_['benchmark']=='10_2') & (data_['setting']=='disjoint')]
#data_ = data_.loc[(data_['benchmark']=='10_5') & (data_['setting']=='disjoint')]
data_ = data_.loc[(data_['benchmark']=='10_2') & (data_['setting']=='overlap')]
#data_ = data_.loc[(data_['benchmark']=='10_5') & (data_['setting']=='overlap')]

sns.lineplot(
		data=data_, markers=True, dashes=True,
		x="step", y="miou", hue="task_id", style="method",
		palette=cb_colors, style_order=['Ours', 'WILSON'])#,
		#legend=None)

# ours solid, wilson dashed
# nice palette, possibly color blind friendly
# add final miou values somewhere


y_min = 10.0
y_max = 80.0
plt.ylim([y_min, y_max])

#for i_ in np.sort(data_.task_id.unique())[1:-1]:
for i_ in np.sort(data_.task_id.unique()):
	plt.vlines(i_, y_min, y_max, linestyles ="dashed", colors ="k", linewidth=0.3)

legend = plt.legend(loc="lower left", edgecolor="black")#, ncol=5)
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((1, 1, 1, 1))

plt.show()


