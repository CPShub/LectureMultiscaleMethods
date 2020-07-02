# Import required modules
import numpy as np                        # numerics
import scipy as sp
from scipy import linalg
import matplotlib.pyplot as plt           # plot environment
from scipy.integrate import odeint        # numerical ODE solver
import ipywidgets as widgets              # interactive widgets
from ipywidgets import interactive, interact
from IPython.display import Javascript    # For external cell execution
import sys

# Initialize function for magic function "matplotlib"
display(Javascript("Jupyter.notebook.execute_cells([23])"))

# Define variable vor switching between static / interactive
global notebook_is_interactive, F11, F22, F12, F21

def notebook_static():
    interactive_plot(True)
    interactive_plot(False)
    global notebook_is_interactive
    notebook_is_interactive=False
    display(Javascript("Jupyter.notebook.execute_cells([10])"))
#    display(Javascript("Jupyter.notebook.execute_cells(" + str(cell_update) + ")"))
    
def notebook_interactive():
    interactive_plot(True)
    global notebook_is_interactive
    notebook_is_interactive=True
    display(Javascript("Jupyter.notebook.execute_cells([10])"))
#    display(Javascript("Jupyter.notebook.execute_cells(" + str(cell_update) + ")"))
	
	
# Add subfolders to working directory
sys.path.insert(0,"./code/02_kinematics")

# Define cells which are updated when changing the values of the deformation gradient
cell_update=[13,18,20]
	
# Print options for numpy, decimal places numpy output
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def F_decomposition_eval(F_11,F_22,F_12,F_21):

    global F11, F22, F12, F21
	
	# local values from widget => global values
    F11=F_11
    F22=F_22
    F12=F_12
    F21=F_21

    F=np.eye(3)
    F[0,0]=F11
    F[1,1]=F22
    F[0,1]=F12
    F[1,0]=F21
	

    #	Polar Decomposition
    R,U=sp.linalg.polar(F, side='right')
    
    #	Deviatoric / Volumetric Decomposition
    F_dv=np.matmul(U,np.eye(3))
    F_d=np.matmul(U,np.eye(3))/np.linalg.det(U)**(1/3)
    F_dr=np.matmul(R,F_d)
    F_drv=F_dr*np.linalg.det(U)**(1/3)
    J=np.linalg.det(U)
	
    print('Deformation Gradient')
    print()
    print('F =')
    print(F)
    print()
    print()
    print('Rotation Tensor')
    print()
    print('R =')
    print(R)
    print()
    print()
    print('Right Stretch Tensor')
    print()
    print('U =')
    print(U)
    print()
    print()
    print('Volume Ratio')
    print()
    print('J =')
    print("%8.3f" % (J))
	
    fig,axs=plt.subplots()
    axs.axis('equal')
    plt.plot([0,1,1,0,0],[0,0,1,1,0], linewidth=3, label='random diagonal')
    plt.plot([0,F_dv[0,0],F_dv[0,0]+F_dv[0,1],F_dv[0,1],0],[0,F_dv[1,0],F_dv[1,0]+F_dv[1,1],F_dv[1,1],0], linewidth=2, label='random diagonal', linestyle='--')
    plt.plot([0,F_drv[0,0],F_drv[0,0]+F_drv[0,1],F_drv[0,1],0],[0,F_drv[1,0],F_drv[1,0]+F_drv[1,1],F_drv[1,1],0], linewidth=2, label='random diagonal')
    axs.legend(['reference configuration','stretch','... + rotation'],loc='lower left')
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title('Polar Decomposition')
    plt.show()
	
    fig,axs=plt.subplots()
    axs.axis('equal')
    plt.plot([0,1,1,0,0],[0,0,1,1,0], linewidth=3, label='random diagonal')
    plt.plot([0,F_dr[0,0],F_dr[0,0]+F_dr[0,1],F_dr[0,1],0],[0,F_dr[1,0],F_dr[1,0]+F_dr[1,1],F_dr[1,1],0], linewidth=2, label='random diagonal', linestyle='--')
    plt.plot([0,F_drv[0,0],F_drv[0,0]+F_drv[0,1],F_drv[0,1],0],[0,F_drv[1,0],F_drv[1,0]+F_drv[1,1],F_drv[1,1],0], linewidth=3, label='random diagonal')
    axs.legend(['reference configuration','deviatoric stretch + rotation','... + change of volume'],loc='lower left')
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title('Volumetric Decomposition')
    plt.show()
	
    display(Javascript("Jupyter.notebook.execute_cells(" + str(cell_update) + ")"))

#	F_11 etc.: local quantity for widget
#	F11  etc.: global quantity
def F_decomposition():
    global F11, F22, F12, F21
    if notebook_is_interactive:
        stress_tensors_eval_interactive = widgets.interactive(F_decomposition_eval,
            {'manual':True},
            F_11=widgets.FloatSlider(min=0.5, max=2, step=0.1, value=1.1),
            F_22=widgets.FloatSlider(min=0.5, max=2, step=0.1, value=1.1),
            F_12=widgets.FloatSlider(min=-0.60001, max=0.60001, step=0.1, value=0.3),
            F_21=widgets.FloatSlider(min=-0.60001, max=0.60001, step=0.1, value=0.0))
        display(stress_tensors_eval_interactive)
    else:
        F11=1.1
        F22=F11
        F12=0.3
        F21=0.0
        F_decomposition_eval(F11,F22,F12,F21)
		
def strain_measures(F11,F22,F12,F21):
    
    F=np.eye(3)
    F[0,0]=F11
    F[1,1]=F22
    F[0,1]=F12
    F[1,0]=F21
	
    #	Strain Tensors
    b=np.matmul(F,np.transpose(F))
    E=0.5*(np.matmul(np.transpose(F),F)-np.identity(3))
    e=0.5*(np.identity(3)-np.linalg.inv(b))
    e_lin=0.5*(F+np.transpose(F))-np.identity(3)
    print('Green-Lagrange Strain Tensor')
    print('E =')
    print(E)
    print()
    print()
    print('Linearized Strain Tensor')
    print('E_lin =')
    print(e_lin)
	

def strain_measures_lin_plot():
    n_max=50
    
    E_shear=np.zeros([3,3,n_max])
    E_tension=np.zeros([3,3,n_max])
    E_lin_shear=np.zeros([3,3,n_max])
    E_lin_tension=np.zeros([3,3,n_max])
    
    
    tension_max=1.5
    shear_max=0.1
    F_tension_ges=np.linspace(1,tension_max,n_max)
    F_shear_ges=np.linspace(0,shear_max,n_max)
    
    for i in list(range(0,n_max,1)):
        F_tension=np.eye(3)
        F_tension[0,0]=F_tension_ges[i]
        F_shear=np.eye(3)
        F_shear[0,1]=F_shear_ges[i]
        F_shear[1,0]=F_shear[0,1]
        #	Strain Tensors
        E_shear[:,:,i]=0.5*(np.matmul(np.transpose(F_shear),F_shear)-np.identity(3))
        E_tension[:,:,i]=0.5*(np.matmul(np.transpose(F_tension),F_tension)-np.identity(3))
        E_lin_shear[:,:,i]=0.5*(F_shear+np.transpose(F_shear))-np.identity(3)
        E_lin_tension[:,:,i]=0.5*(F_tension+np.transpose(F_tension))-np.identity(3)
    	
    
    plt.xlabel("$F_{11}$")
    plt.ylabel("11-Component")
    plt.title("Linearization for Uniaxial Tension")
    plt.plot(F_tension_ges,E_tension[0,0,:],F_tension_ges,E_lin_tension[0,0,:])
    plt.gca().legend(('Green-Lagrange Strain Tensor','Linearized Strain Tensor'))
    plt.show()
    
    #fig,axs=plt.subplots()
    #axs.axis('equal')
    #plt.xlabel("$I_1(C)$")
    #plt.ylabel("12-Component")
    #plt.plot(F_shear_ges,E_shear[0,0,:],F_shear_ges,E_lin_shear[0,0,:]);
    #plt.gca().legend(('Green-Lagrange Strain Tensor','Linearized Strain Tensor'));
    #plt.ylim(-0.001,0.015);
    #plt.show()

	
def eigenvalues(F11,F22,F12,F21):
    F=np.eye(3)
    F[0,0]=F11
    F[1,1]=F22
    F[0,1]=F12
    F[1,0]=F21
	
    #	Polar Decomposition
    R,U=sp.linalg.polar(F, side='right')

    #	Eigenvalues
    lam_U, eigv_U = np.linalg.eig(U)
	
    #	Deviatoric / Volumetric Decomposition
    F_dv=np.matmul(U,np.eye(3))
    F_d=np.matmul(U,np.eye(3))/np.linalg.det(U)**(1/3)
    F_dr=np.matmul(R,F_d)
    F_drv=F_dr*np.linalg.det(U)**(1/3)
    J=np.linalg.det(U)
    
    fig,axs=plt.subplots()
    
    
    plt.plot([0,1,1,0,0],[0,0,1,1,0])
    plt.plot([0,F_drv[0,0],F_drv[0,0]+F_drv[0,1],F_drv[0,1],0],[0,F_drv[1,0],F_drv[1,0]+F_drv[1,1],F_drv[1,1],0], linewidth=2,linestyle='--')
    
    plt.arrow(0, 0, lam_U[0]*eigv_U[0,0], lam_U[0]*eigv_U[1,0], length_includes_head=True,
              head_width=0.1, head_length=0.1,color='red')
    plt.arrow(0, 0, lam_U[1]*eigv_U[0,1], lam_U[1]*eigv_U[1,1], length_includes_head=True,
              head_width=0.1, head_length=0.1,color='red')
    		  
    plt.plot(0, 0, lam_U[0]*eigv_U[0,0], lam_U[0]*eigv_U[1,0],color='red')
    plt.plot(0, 0, lam_U[1]*eigv_U[0,1], lam_U[1]*eigv_U[1,1],color='red')
    
    
    		  
    ax = plt.gca()
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title('Principal Stretches')
    
    
    
    axs.axis('equal')
    plt.show()
    
    print('Principal Stretches')
    print(lam_U)
    print()
    print()
    print('Eigenvectors (ordered in columns)')
    print(eigv_U)
	
	
def deformation_state(F11,F22,F12,F21):
    F=np.eye(3)
    F[0,0]=F11
    F[1,1]=F22
    F[0,1]=F12
    F[1,0]=F21
    b=np.matmul(F,np.transpose(F))
    b_unimod=b/np.linalg.det(b)**(1/3)
    lam_quad, eigv_E = np.linalg.eig(b_unimod)
    
    
    I1=lam_quad[0]+lam_quad[1]+lam_quad[2]
    I2=lam_quad[0]*lam_quad[1]+lam_quad[0]*lam_quad[2]+lam_quad[1]*lam_quad[2] 
    
    xmax=max(I1-3,I2-3)*2+3.01
    
    x_tension=np.linspace(3,xmax,100)
    y_tension=4*(x_tension/3)**0.5*np.cos(1/3*np.arccos(-3*(3/x_tension**3.0)**0.5))+3/(4*x_tension*(np.cos((1/3)*np.arccos(-3*(3/x_tension**3)**0.5)))**2)
    
    
    fig,axs=plt.subplots()
    plt.xlim([3,xmax])
    plt.ylim([3,xmax])
    plt.title("State of Deviatoric Deformation")
    plt.xlabel("$I_1$")
    plt.ylabel("$I_2$")
    plt.gca().set_aspect('equal')
    plt.plot(x_tension,y_tension)
    plt.plot(y_tension,x_tension)
    plt.plot(x_tension,x_tension)
    plt.plot([I1], [I2], marker='o', markersize=8, color="red")
    plt.gca().legend(('Uniaxial Tension','Equibiaxial Tension','Shear'))
    plt.show()
		
		
		
		
# Initialize notebook with static content    
display(Javascript("Jupyter.notebook.execute_cells([5])"))

# Run cell for "strain_measure_lin_plot"  
display(Javascript("Jupyter.notebook.execute_cells([15])"))