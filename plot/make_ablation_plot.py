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
sns.set_palette(sns.color_palette(cb_colors)) # not working, not sure why


data_ = pd.read_csv('./wsci-sis-ablations-gains.csv')

print(data_)

data_ =  data_[196:].iloc[:,:9]

data_.rename(columns={
		'\\tau ablation':'param_type', 'plot':'param_value', 'Unnamed: 2':'miou_value',
		'Unnamed: 3':'miou_type', 'Unnamed: 4':'dataset',
		'Unnamed: 5':'benchmark', 'Unnamed: 6':'setting', 'Unnamed: 7':'method'}, inplace=True )

data_ = data_.astype({
		'param_type':'str','param_value':'float', 'miou_value':'float',
		'miou_type':'str', 'dataset':'str',
		'benchmark':'str', 'setting': 'str', 'method': 'str'})

print(data_)

# only keep one ablation
data_ = data_[data_.param_type=='lambda']
data_ = data_[data_.miou_type!='all']
#data_ = data_[data_.param_type=='lambda']

#data_ = data_.loc[(data_['benchmark']=='10_2') & (data_['setting']=='disjoint')]
#data_ = data_.loc[(data_['benchmark']=='10_5') & (data_['setting']=='disjoint')]
#data_ = data_.loc[(data_['benchmark']=='10_2') & (data_['setting']=='overlap')]
#data_ = data_.loc[(data_['benchmark']=='10_5') & (data_['setting']=='overlap')]

data_ = data_.loc[(data_['benchmark']=='10_2')]

fig, ax = plt.subplots(1,1)

#sns.pointplot(
#		data=data_,# markers=True, dashes=True,
#		x="param_value", y="miou_all",
#		color='#882255', ax=ax)#,
#		#legend=None)

g1 = sns.lineplot(
		data=data_,# markers=True, dashes=True,
		x="param_value", y="miou_value",
		ax=ax, hue="miou_type", style_order=['overlap', 'disjoint'],
		style="setting", markers=True)#, '-.'])

#lines1 = g1.get_lines()
#[l.set_color('#117733') for l in lines1]
#[l.set_markerfacecolor('#117733') for l in lines1]

#g2 = sns.pointplot(
#		data=data_,# markers=True, dashes=True,
#		x="param_value", y="miou_old",
#		color='#332288', ax=ax, hue="setting",
#		linestyles=['-', ':'])#, '-.'])
#
#lines2 = g2.get_lines()
#[l.set_color('#332288') for l in lines2]
#[l.set_markerfacecolor('#332288') for l in lines2]

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

h_1 = mpatches.Patch(color='#117733', label='miou_new')
h_2 = mpatches.Patch(color='#332288', label='miou_old')
h_3 = Line2D([0], [0], color='k', lw=1, label='overlap', linestyle='-')
h_4 = Line2D([0], [0], color='k', lw=1, label='disjoint', linestyle='--')

plt.legend(handles=[h_1, h_2, h_3, h_4])

#plt.xticks([1,3,5,7,10])
#plt.xticks([0.1,1,10])

#y_min = 0.0
#y_max = 50.0
#plt.ylim([y_min, y_max])

#for i_ in np.sort(data_.task_id.unique())[1:-1]:
#for i_ in np.sort(data_.task_id.unique()):
#	plt.vlines(i_, y_min, y_max, linestyles ="dashed", colors ="k", linewidth=0.3)

plt.xscale('log')

plt.show()


