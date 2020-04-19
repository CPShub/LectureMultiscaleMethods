# Solve and Plot Michaelis Menten ODE

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt	


def plot(xs,xsmac,ysmic,ysmac,y_1_0,y_2_0,kap,lam,eps):
    print('y_1_0: %4.2f, y_2_0: %4.2f' % (y_1_0,y_2_0))
    print('kappa: %4.2f, lambda: %4.2f' % (kap,lam))
    print('epsilon: %4.2f' % (eps))
# plot y(t) over t, micro solution
    plt.xlabel("$t$")
    plt.ylabel("$y(t)$")
    plt.title("Michaelis Menten ODE")
    plt.plot(xs,ysmic[:,0]);
    plt.plot(xs,ysmic[:,1]);
    plt.gca().legend(('$y_1$','$y_2$'))
    plt.show()
# plot y_2 over y_1
    plt.xlabel("$y_1(t)$")
    plt.ylabel("$y_2(t)$")
# micro solution
    plt.plot(ysmic[:,0],ysmic[:,1]);
# macro solution
    plt.plot(xsmac,ysmac,'-.');
    plt.gca().legend(('$micro$','$macro$'))
    plt.show()


	
def solve(param): 
    y_1_0=param[0]
    y_2_0=param[1]
    kap=param[2]
    lam=param[3]
    eps=param[4]
# numeric solution for micro model  
    def dy_dx(y, x):
       return [-y[0]+(y[0]+kap-lam)*y[1], 1/eps*(y[0]-(y[0]+kap)*y[1])] 		  
    y0 = [y_1_0, y_2_0]
    xs = np.linspace(0, 1, 200)
    ysmic = odeint(dy_dx, y0, xs)
# analytic solution for macro model
    xsmac=np.linspace(np.amin(ysmic[:,0]),np.amax(ysmic[:,0]),200)
    ysmac = xsmac / (xsmac + kap)  
# plot parameters and solution
    plot(xs,xsmac,ysmic,ysmac,y_1_0,y_2_0,kap,lam,eps)
