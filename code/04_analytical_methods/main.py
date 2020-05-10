#%% Import 

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from scipy.optimize import fsolve

#%% Bounds

def bV(c1s,c2s,f2):
    return np.array([1-f2,f2])@np.array([c1s,c2s])

def bR(c1s,c2s,f2):
    return 1/(np.array([1-f2,f2])@np.array([1/c1s,1/c2s]))

#%% Eshelby's solution

def aSIP(cIs,cMs):
    a1 = (cMs[0]+2*cMs[1])/(cIs[0]+2*cMs[1])
    a2 = (5*cMs[1]*(cMs[0]+2*cMs[1]))/(
        2*cIs[1]*(cMs[0]+3*cMs[1])+cMs[1]*(3*cMs[0]+4*cMs[1])
        )
    return np.array([a1,a2])

#%% Approximations

# Vector representation of IS for iso(1,1)
IS = np.array([1,1])

# General
def app(c1s,c2s,f2,a):
    return c1s + f2*(c2s-c1s)*a

# DD
def appDD1(c1s,c2s,f2):
    a = aSIP(cIs=c2s,cMs=c1s)
    return app(c1s,c2s,f2,a)

def appDD2(c1s,c2s,f2):
    aS = aSIP(cIs=c1s,cMs=c2s)
    a = (IS-(1-f2)*aS)
    return app(c1s,c2s,1,a)

# MT
def appMT1(c1s,c2s,f2):
    aS = aSIP(cIs=c2s,cMs=c1s)
    a = 1/((1-f2)*1/aS+f2*IS)
    return app(c1s,c2s,f2,a)

def appMT2(c1s,c2s,f2):
    aS = aSIP(cIs=c1s,cMs=c2s)
    a = 1/((1-f2)*aS+f2*IS)
    return app(c1s,c2s,f2,a)

# SC1
def appSC1(c1s,c2s,f2):
    def eqs(ceffs):
        a = aSIP(cIs=c2s,cMs=ceffs)
        ceffs2 = app(c1s,c2s,f2,a)
        return ceffs - ceffs2
    initial_guess = (appMT1(c1s, c2s, f2) + appMT2(c1s, c2s, f2))/2
    return np.array(fsolve(eqs,initial_guess))

#%% Plot

def plot(c1s,c2s,orientation='h'):
    f2s = np.linspace(0,1,20)
    
    p = np.array([
        [f(c1s,c2s,f2) for f2 in f2s] 
        for f in [bV,bR,appDD1,appDD2,appMT1,appMT2,appSC1]
        ])
    
    if orientation=='h':
        fig,ax = plt.subplots(1,2,figsize=(10,4))
    else:
        fig,ax = plt.subplots(2,1,figsize=(5,8))
    
    y_labels = ['$c^*_1 = 3K^*$','$c^*_2 = 2G^*$']
    for i in [0,1]:
        ax[i].plot(f2s,p[0,:,i],label='Voigt',color='red')
        ax[i].plot(f2s,p[1,:,i],label='Reuss',color='red',linestyle='--')
        ax[i].plot(f2s,p[2,:,i],label='DD1',color='blue')
        ax[i].plot(f2s,p[3,:,i],label='DD2',color='blue',linestyle='--')
        ax[i].plot(f2s,p[4,:,i],label='MT1',color='orange')
        ax[i].plot(f2s,p[5,:,i],label='MT2',color='orange',linestyle='--')
        ax[i].scatter(f2s,p[6,:,i],label='SC1',color='green')
        ax[i].set_ylim([0,100])
        ax[i].set_xlabel('$f^{(2)}$')
        ax[i].set_ylabel(y_labels[i])
        ax[i].legend()
    plt.show()
    
#%% Static plot

def show_static_plot():    
    c1s = np.array([10,100])
    c2s = np.array([80,30])   
    print('Static plot')
    print('Material 1 eigenvalues:')
    print(c1s)
    print('Material 2 eigenvalues:')
    print(c2s)
    plot(c1s,c2s)

#%% Interactive plot

# Interactive variables
c11 = widgets.IntSlider(1,min=1,max=100,description='$c^{(1)}_1 = 3K^{(1)}$')
c12 = widgets.IntSlider(1,min=1,max=100,description='$c^{(1)}_2 = 2G^{(1)}$')
c21 = widgets.IntSlider(100,min=1,max=100,description='$c^{(2)}_1 = 3K^{(2)}$')
c22 = widgets.IntSlider(100,min=1,max=100,description='$c^{(2)}_2 = 2G^{(2)}$')

c11s = widgets.Dropdown(options = [1,10,50,100], value = 1, description='$3K^{(1)}$')
c12s = widgets.Dropdown(options = [1,10,50,100], value = 1, description='$2G^{(1)}$')
c21s = widgets.Dropdown(options = [1,10,50,100], value = 100, description='$3K^{(2)}$')
c22s = widgets.Dropdown(options = [1,10,50,100], value = 100, description='$2G^{(2)}$')

# Interactive devices
devices = widgets.Dropdown(
    options = [('',0),('computer',1),('smartphone',2)]
    ,description = 'Select device'
)

def plot_computer(c11,c12,c21,c22):
    plot(np.array([c11,c12]),np.array([c21,c22]))
    
def plot_smartphone(c11,c12,c21,c22):
    plot(np.array([c11,c12]),np.array([c21,c22]),orientation='v')

def check(device):
    if device==0:
        print('Please choose a device.')
    if device==1:
        head1 = widgets.HBox([widgets.Label('Material 1'),c11,c12])
        head2 = widgets.HBox([widgets.Label('Material 2'),c21,c22])
        bottom = widgets.interactive(
            plot_computer
            ,{'manual':True}
            ,c11=c11,c12=c12,c21=c21,c22=c22
        )
        bottom.children[-2].description = 'Run/Update'
        display(head1)
        display(head2)
        display(bottom.children[-2])
        display(bottom.children[-1])
    if device==2:
        head = widgets.VBox([
            c11s,c12s
            ,c21s,c22s
            ])
        bottom = widgets.interactive(
            plot_smartphone
            ,{'manual':True}
            ,c11=c11s,c12=c12s,c21=c21s,c22=c22s
        )
        bottom.children[-2].description = 'Run/Update'
        display(head)
        display(bottom.children[-2])
        display(bottom.children[-1])
    
check_w = widgets.interactive(
    check
    ,device=devices
)
    
def start_interactive():
    display(check_w)