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


data_ = pd.read_csv('./wsci-sis-ablations.csv')

data_ =  data_[105:-3].iloc[:,:9]


print(data_)
data_.rename(columns={
		'\\tau ablation':'param_type', 'Unnamed: 1':'param_value', 'Unnamed: 2':'miou_old',
		'Unnamed: 3':'miou_new', 'Unnamed: 4':'miou_all', 'Unnamed: 5':'dataset',
		'Unnamed: 6':'benchmark', 'Unnamed: 7':'setting', 'Unnamed: 8':'method'}, inplace=True )


data_ = data_.astype({
		'param_type':'str','param_value':'float', 'miou_old':'float',
		'miou_new':'float', 'miou_all':'float', 'dataset':'str',
		'benchmark':'str', 'setting': 'str', 'method': 'str'})

# only keep one ablation
data_ = data_[data_.param_type=='tau']

#data_ = data_.loc[(data_['benchmark']=='10_2') & (data_['setting']=='disjoint')]
#data_ = data_.loc[(data_['benchmark']=='10_5') & (data_['setting']=='disjoint')]
data_ = data_.loc[(data_['benchmark']=='10_2') & (data_['setting']=='overlap')]
#data_ = data_.loc[(data_['benchmark']=='10_5') & (data_['setting']=='overlap')]

fig, ax = plt.subplots(1,1)

sns.pointplot(
		data=data_,# markers=True, dashes=True,
		x="param_value", y="miou_all",
		color='#332288', ax=ax)#,
		#legend=None)

sns.pointplot(
		data=data_,# markers=True, dashes=True,
		x="param_value", y="miou_new",
		color='#117733', ax=ax)#,
		#legend=None)

sns.pointplot(
		data=data_,# markers=True, dashes=True,
		x="param_value", y="miou_old",
		color='#882255', ax=ax)#,
		#legend=None)



# ours solid, wilson dashed
# nice palette, possibly color blind friendly
# add final miou values somewhere


y_min = 0.0
y_max = 50.0
plt.ylim([y_min, y_max])

#for i_ in np.sort(data_.task_id.unique())[1:-1]:
#for i_ in np.sort(data_.task_id.unique()):
#	plt.vlines(i_, y_min, y_max, linestyles ="dashed", colors ="k", linewidth=0.3)

legend = plt.legend(loc="lower left", edgecolor="black")#, ncol=5)
legend.get_frame().set_alpha(None)
legend.get_frame().set_facecolor((1, 1, 1, 1))

plt.show()


