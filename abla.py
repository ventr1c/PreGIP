#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# %%
matplotlib.rcParams.update({'font.size': 30, 'font.family': 'serif'})
labels = ['Fix','Finetune']
width = 0.3       # the width of the bars: can also be len(x) sequence
x =np.arange(len(labels))

GIP = [100,94.5]


GIN_N = [100, 93.2]
GIN_S = [99, 77]

GIN_F = [100, 70]


fig, ax = plt.subplots(figsize=(10,8))

ax.bar(x-3*width/4,GIP,2*width/4,label="PreGIP")
ax.bar(x-1*width/4,GIN_N,2*width/4,label="PreGIP\\N")
ax.bar(x+1*width/4,GIN_S,2*width/4,label="PreGIP\\S")
ax.bar(x+3*width/4,GIN_F,2*width/4,label="PreGIP\\F")

ax.set_xticks([0,1])
ax.set_xticklabels(labels,fontsize=40)
# ax.xticks([],)
plt.yticks([68,78,88,98],fontsize=30)
# plt.set_xticklabels(fontsize=50)
ax.set_ylabel('IP ROC (%)',fontsize=40)
# ax.set_title('Scores by group and gender')
ax.legend(ncol=2,fontsize=25)
plt.ylim(67,111)
plt.tight_layout()
# plt.show()
plt.savefig("abla_IP.pdf")
# %%

labels = ['Fix','Finetune']
width = 0.3       # the width of the bars: can also be len(x) sequence
x =np.arange(len(labels))

GIP = [70.3,70.1]
GIP_err = [1.1,0.4]


GIN_N = [64.6, 67.1]
GIN_N_err = [2,1.2]
GIN_S = [64.1, 68.5]
GIN_S_err = [2,2.1]

GIN_F = [70.1, 69.8]
GIN_F_err = [1.7, 1.5]


fig, ax = plt.subplots(figsize=(10,8))

ax.bar(x-3*width/4,GIP,2*width/4,yerr=GIP_err,error_kw={"elinewidth":5},label="PreGIP")
ax.bar(x-1*width/4,GIN_N,2*width/4,yerr=GIN_N_err,error_kw={"elinewidth":5},label="PreGIP\\N")
ax.bar(x+1*width/4,GIN_S,2*width/4,yerr=GIN_S_err,error_kw={"elinewidth":5},label="PreGIP\\S")
ax.bar(x+3*width/4,GIN_F,2*width/4,yerr=GIN_F_err,error_kw={"elinewidth":5},label="PreGIP\\F")

ax.set_xticks([0,1])
ax.set_xticklabels(labels,fontsize=40)
# ax.xticks([],)
# plt.yticks([68,78,88,98],fontsize=30)
# plt.set_xticklabels(fontsize=50)
ax.set_ylabel('Accuracy (%)',fontsize=40)
# ax.set_title('Scores by group and gender')
ax.legend(ncol=2,fontsize=25)
plt.ylim(58,76)
plt.tight_layout()
# plt.show()
plt.savefig("abla_ACC.pdf")

# %%
