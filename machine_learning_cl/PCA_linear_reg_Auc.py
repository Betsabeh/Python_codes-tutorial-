import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp

#------------------------------------------------------
data = pd.read_csv('https://archive.ics.uci.edu/ml/'
            'machine-learning-databases/wine/wine.data',header=None)

X = data.loc[:, 1:].values
Y = data.loc[:, 0].values
#print(Y)
#make pipe line
model=make_pipeline(StandardScaler(), PCA(n_components=2),
                    LogisticRegression(solver='lbfgs'))


cv=list(StratifiedKFold(5).split(X,Y))

fig = plt.figure(figsize=(5,7))
all_TPR = []
mean_TPR = 0.0
mean_FPR = np.linspace(0, 1, 100)

for i, (train, test) in enumerate(cv):
    model = model.fit(X[train],Y[train])
    prob = model.predict_proba(X[test])
    #print(prob)
    fpr, tpr, thresholds = roc_curve(Y[test],prob[:,0],pos_label=1)
    mean_TPR += interp(mean_FPR, fpr, tpr)
    mean_TPR[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr,tpr,label = 'ROC fold %d (area = %0.2f)'% (i + 1, roc_auc))

plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), label='Random guessing')
mean_TPR /= len(cv)
mean_TPR[-1] = 1.0
mean_auc = auc(mean_FPR, mean_TPR)
plt.plot(mean_FPR, mean_TPR, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
plt.plot([0, 0, 1],[0, 1, 1],linestyle=':', color='black',label='Perfect performance')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend(loc="lower right")
plt.show()




