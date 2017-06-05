# -*- coding: utf-8 -*-

import numpy as np

import matplotlib as mplt
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import matplotlib
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
# matplotlib.style.use('ggplot')
import itertools

def y_distribution_plot( dta, valset ):
    
    y_pos = np.arange(len( valset ))
    cnt=[0 for _ in valset]
    tot_cnt= len(dta)
    
    for i in dta:
        cnt[int(i)]= cnt[int(i)]+1 

    for i in range(len(valset)):
        cnt[i]= cnt[i]*1.0/tot_cnt

    fig = plt.figure()
    plt.bar(y_pos, cnt, align='center', alpha=0.5)
    plt.xticks(y_pos, valset, rotation=70)
#   plt.ylim([0,0.3])
    plt.ylabel('% of data instances w.r.t. \n each label')
    plt.xlabel('Labels')
#   plt.title('Age label')
#   fig.savefig('./results/classDis.jpg', format='jpg', bbox_inches='tight')
    