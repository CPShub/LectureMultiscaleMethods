# Solve and Plot Michaelis Menten ODE

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt	
	
	
def eval_y2_0(solution_old,param,Add):    
    solution=solve(param)
    if Add == False:
        y_1_ges=solution[2][:,0]
        y_2_ges=solution[2][:,1]
        y_2_0_ges=np.array([param[1]])
    else:   
        y_1_ges=np.row_stack((solution_old[0],solution[2][:,0]))
        y_2_ges=np.row_stack((solution_old[1],solution[2][:,1]))
        y_2_0_ges=np.row_stack((solution_old[2],[param[1]]))
# analytic solution for macro model
    kap=param[2]
    xsmac=np.linspace(np.amin(y_1_ges),np.amax(y_1_ges),200)
    ysmac = xsmac / (xsmac + kap)  
    plot_y2_0(y_1_ges,y_2_ges,y_2_0_ges,xsmac,ysmac) 
    return y_1_ges,y_2_ges,y_2_0_ges

	
def plot_y2_0(x_ges,y_ges,y_2_0_ges,xsmac,ysmac):
    plt.plot(xsmac,ysmac,'-.',label='macro')
    if x_ges.ndim == 1:
        plt.plot(x_ges,y_ges,label=str(y_2_0_ges))
    else:
        for i in range (0,x_ges.shape[0]):
            plt.plot(x_ges[i],y_ges[i],label=str(y_2_0_ges[i]))	
    plt.xlabel("$y_1(t)$")
    plt.ylabel("$y_2(t)$")
    plt.title("Variation of $y_{2,\;0}$")
    plt.legend()
    plt.show()
	

def plot_y1_y2(solution,param):
    xs=solution[0]
    xsmac=solution[1]
    ysmic=solution[2]
    ysmac=solution[3]
    y_1_0=param[0]
    y_2_0=param[1]
    kap=param[2]
    lam=param[3]
    eps=param[4]
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
    return xs, xsmac, ysmic, ysmac
	