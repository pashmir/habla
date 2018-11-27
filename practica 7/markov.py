import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.linalg import sqrtm
from matplotlib import colors as mcolors

# Specification of some HMMs for classification

class hmm:
    
    def __init__(self,means,vrs,trans):
        self.means = means
        self.vrs = vrs
        self.trans = trans
        
        self.dim = means.shape[1]
        self.numStates = means.shape[0] + 2
        self.devs = np.array([sqrtm(M) for M in vrs])
     
    
def genhmm(hmm):

    stateSeq = [0]
    while stateSeq[-1] != hmm.numStates-1:
        stateSeq.append(int(np.random.choice(hmm.numStates, 1, p=hmm.trans[stateSeq[-1],:])))

    stateSeq = np.array(stateSeq)
    x = np.add.reduce(np.matmul(np.random.randn(stateSeq.size-2, hmm.dim)[:,np.newaxis,:],hmm.devs[(stateSeq[1:-1]-1)]),axis=1)
    
    return x, stateSeq

def plotseq(hmm, stateSeq, x):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    classes_names = np.array(['Fonema '+str(i+1) for i in range(hmm.numStates-2)],dtype=object)
    colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).values())

    ax1.plot(x[:,0])
    ax2.plot(x[:,1])

    s = stateSeq[1:-1]-1
    for i in range(hmm.numStates-2):
        mask = s==i
        ax1.scatter(np.arange(s.size)[mask],x[mask,0],color=colors[i],label=classes_names[i])
        ax2.scatter(np.arange(s.size)[mask],x[mask,1],color=colors[i],label=classes_names[i])

    ax1.legend()
    
    return ax1, ax2
    
    
def plotseq2(hmm,stateSeq,x,gauss=False):
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    classes_names = np.array(['Fonema '+str(i+1) for i in range(hmm.numStates-2)],dtype=object)
    colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).values())

    ax.plot(x[:,0],x[:,1])
    
    for i in range(x.shape[0]):
        ax.annotate(str(i+1),(x[i,0],x[i,1]))
    

    s = stateSeq[1:-1]-1
    for i in range(hmm.numStates-2):
        mask = s==i
        ax.scatter(x[mask,0],x[mask,1],color=colors[i],label=classes_names[i])
        if gauss:
            covariance_ellipse(hmm.means[i],hmm.vrs[i],ax=ax,color=colors[i])
        
    ax.legend()
    
    return ax
    
    
def covariance_ellipse(mu, sigma, ax=None, color="k"):
    
    # Cálculo de los autovalores:
    vals, vecs = np.linalg.eigh(sigma)
    
    # Inclinación de la elipse:
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Semiejes de la elipse:
    w, h = 2 * np.sqrt(vals)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
                
    ax.tick_params(axis='both', which='major', labelsize=20)
    ellipse = Ellipse(mu, w, h, theta, color=color)
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse)
    
    return ax