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
from matplotlib import interactive as mat_int		  # Interactive plots
interactive(True)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Initialize function for magic function "matplotlib"
display(Javascript("Jupyter.notebook.execute_cells([24])"))

# Define variable vor switching between static / interactive
global notebook_is_interactive

def notebook_static():
    interactive_plot(True)
    interactive_plot(False)
    global notebook_is_interactive
    notebook_is_interactive=False
    display(Javascript("Jupyter.notebook.execute_cells(" + str(cell_update) + ")"))
    
def notebook_interactive():
    interactive_plot(True)
    global notebook_is_interactive
    notebook_is_interactive=True
    display(Javascript("Jupyter.notebook.execute_cells(" + str(cell_update) + ")"))

	
# Add subfolders to working directory
sys.path.insert(0,"./code/03_kinetics")

# Define cells which are updated when switching between static / interactive
cell_update=[10,12,15,18,21]
	
# Print options for numpy, decimal places numpy output
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


# Evaluation and output for plane stress 
def stress_tensors_eval(F11,F22,F33,F12,F21):
    
    F=np.eye(3)
    F[0,0]=F11
    F[1,1]=F22
    F[2,2]=F33
    F[0,1]=F12
    F[1,0]=F21

    b=np.matmul(F,np.transpose(F))
    E=0.5*(np.matmul(np.transpose(F),F)-np.identity(3))
    e=0.5*(np.identity(3)-np.linalg.inv(b))
    e_lin=0.5*(F+np.transpose(F))-np.identity(3)

    J=np.linalg.det(F)

    #	Stress Tensors
    #	Parameter
    mu=12.0
    lam=8.0

    S=2.0*mu*E+lam*np.trace(E)*np.identity(3)
    P=np.matmul(F,S)
    sigma=1.0/J*np.matmul(P,np.transpose(F))

    print('Deformation Gradient')
    print()
    print('F =')
    print(F)
    print()
    print()
    print('Second Piola Kirchhoff Tensor')
    print()
    print('S=')
    print(S)
    print()
    print()
    print('First Piola Kirchhoff Stress Tensor')
    print()
    print('P =')
    print(P)
    print()
    print()
    print('Cauchy Stress Tensor')
    print()
    print('sigma =')
    print(sigma)


    # Eigenvalues
    lam_sig, eigv_sig = np.linalg.eig(sigma)
    lam_S, eigv_S = np.linalg.eig(S)
    

    fig,axs=plt.subplots()

    plt.plot([0,1,1,0,0],[0,0,1,1,0])

    plt.plot([0,F[0,0],F[0,0]+F[0,1],F[0,1],0],[0,F[1,0],F[1,0]+F[1,1],F[1,1],0], linewidth=2,linestyle='--')

    plt.arrow(0, 0, eigv_S[0,0], eigv_S[1,0], length_includes_head=True,
          head_width=0.1, head_length=0.1,color='green')
    plt.arrow(0, 0, eigv_S[0,1], eigv_S[1,1], length_includes_head=True,
          head_width=0.1, head_length=0.1,color='green')
		  
    plt.plot(0, 0, eigv_S[0,0], eigv_S[1,0],color='green',label='Second Piola-Kirchhoff')
    plt.plot(0, 0, eigv_S[0,1], eigv_S[1,1],color='green')

    plt.arrow(0, 0, eigv_sig[0,0],eigv_sig[1,0], length_includes_head=True,
          head_width=0.1, head_length=0.1,color='red')
    plt.arrow(0, 0, eigv_sig[0,1], eigv_sig[1,1], length_includes_head=True,
          head_width=0.1, head_length=0.1,color='red')
		  
    plt.plot(0, 0, eigv_sig[0,0], eigv_sig[1,0],color='red',label='Cauchy')
    plt.plot(0, 0, eigv_sig[0,1], eigv_sig[1,1],color='red')

    ax = plt.gca()
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title('Eigenvectors Stress Tensor')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    axs.axis('equal')
    plt.show()

    print('Eigenvalues S')
    print(lam_S)
    print()
    print('Eigenvectors S (ordered in columns)')
    print(eigv_S)
    print()
    print()
    print('Eigenvalues sigma')
    print(lam_sig)
    print()
    print('Eigenvectors sigma (ordered in columns)')
    print(eigv_sig)
    
def stress_tensors():
    if notebook_is_interactive:
        stress_tensors_eval_interactive = widgets.interactive(stress_tensors_eval,
            {'manual':True},
            F11=widgets.FloatSlider(min=0.5, max=2, step=0.1, value=1),
            F22=widgets.FloatSlider(min=0.5, max=2, step=0.1, value=1),
            F33=widgets.FloatSlider(min=0.5, max=2, step=0.1, value=1),
            F12=widgets.FloatSlider(min=-0.60001, max=0.60001, step=0.1, value=0),
            F21=widgets.FloatSlider(min=-0.60001, max=0.60001, step=0.1, value=0))
        display(stress_tensors_eval_interactive)
    else:
        stress_tensors_eval(1.0,1.0,1.0,0.2,0.0)
        

def stress_tensors_treloar():
    
    fig,axs=plt.subplots()
   
    lam_tension=np.array([1,1.01,1.12,1.24,1.39,1.61,1.89,2.17,2.42,3.01,3.58,4.03,4.76,5.36,5.76,6.16,6.4,6.62,6.87,7.05,7.16,7.27,7.43,7.5,7.61])
    stress_tension_P=np.array([0,0.03,0.14,0.23,0.32,0.41,0.5,0.58,0.67,0.85,1.04,1.21,1.58,1.94,2.29,2.67,3.02,3.39,3.75,4.12,4.47,4.85,5.21,5.57,6.3])
    stress_tension_S=stress_tension_P/lam_tension
    stress_tension_sigma=stress_tension_P*lam_tension

    plt.xlabel("$\lambda \;[\; ]$")
    plt.ylabel("$11-component \; [MPa]$")
    plt.title('Uniaxial Tension')

    plt.plot(lam_tension,stress_tension_S,'--',marker='o',label='S')
    plt.plot(lam_tension,stress_tension_P,'--',marker='o',label='P')
    plt.plot(lam_tension,stress_tension_sigma,'--',marker='o',label='sigma')

    plt.legend()
    plt.show()

    fig,axs=plt.subplots()

    lam_shear=np.array([1,1.06,1.14,1.21,1.32,1.46,1.87,2.4,2.98,3.48,3.96,4.36,4.69,4.96])
    stress_shear_P=np.array([0,0.07,0.16,0.24,0.33,0.42,0.59,0.76,0.93,1.11,1.28,1.46,1.62,1.79])
    stress_shear_S=stress_shear_P/lam_shear
    stress_shear_sigma=stress_shear_P*lam_shear

    plt.xlabel("$\lambda \;[\; ]$")
    plt.ylabel("$11-component \; [MPa]$")
    plt.title('Pure Shear')

    plt.plot(lam_shear,stress_shear_S,'--',marker='o',label='S')
    plt.plot(lam_shear,stress_shear_P,'--',marker='o',label='P')
    plt.plot(lam_shear,stress_shear_sigma,'--',marker='o',label='sigma')

    plt.legend()
    plt.show()
 

def dev_hyd_stress():
    if notebook_is_interactive:
        dev_hyd_stress_eval_interactive = widgets.interactive(dev_hyd_stress_eval,
            {'manual':True},
            sig11=widgets.FloatSlider(min=0, max=2, step=0.1, value=1),
            sig22=widgets.FloatSlider(min=0, max=2, step=0.1, value=1),
            sig33=widgets.FloatSlider(min=0, max=2, step=0.1, value=1),
            sig12=widgets.FloatSlider(min=0, max=2, step=0.1, value=1),
            sig13=widgets.FloatSlider(min=0, max=2, step=0.1, value=1),
            sig23=widgets.FloatSlider(min=0, max=2, step=0.1, value=1))
        display(dev_hyd_stress_eval_interactive)
    else:
        dev_hyd_stress_eval(1.0,0.0,0.0,1.0,0.0,0.0)
        

def dev_hyd_stress_eval(sig11,sig22,sig33,
                   sig12,sig13,sig23):
    sigma=np.eye(3)
    sigma[0,0]=sig11
    sigma[1,1]=sig22
    sigma[2,2]=sig33
    sigma[0,1]=sig12
    sigma[0,2]=sig13
    sigma[1,2]=sig23

    sigma[1,0]=sigma[0,1]
    sigma[2,0]=sigma[0,2]
    sigma[2,1]=sigma[1,2]

    p=np.trace(sigma)/3.0

    sigma_dev=sigma-p*np.identity(3)
    
    sig_main, v = np.linalg.eig(sigma)

    sigma_principal_axes=np.zeros([3,3])
    sigma_principal_axes[0,0]=sig_main[0]
    sigma_principal_axes[1,1]=sig_main[1]
    sigma_principal_axes[2,2]=sig_main[2]
    

    print('Cauchy Stress Tensor')
    print()
    print('sigma =')
    print(sigma)
    print()
    print()
    print('sigma_principal_axes =')
    print(sigma_principal_axes)
    print()
    print()
    print('Hydrostatic Cauchy Stress Tensor')
    print()
    print('p*I =')
    print(p*np.identity(3))
    print()
    print()
    print('Deviatoric Cauchy Stress Tensor')
    print()
    print('sigma_dev =')
    print(sigma_dev)
            
        
    
    fig,axs=plt.subplots()
    ax = plt.axes(projection='3d')

    # Hydrostatic stress
    p_stress = np.array([0, p])
    ax.plot3D(p_stress, p_stress, p_stress, 'red',label='Hydrostatic Stress')


    # Total stress
    x_tot = np.array([0, sig_main[0]])
    y_tot = np.array([0, sig_main[1]]) 
    z_tot = np.array([0,sig_main[2]])   
    ax.plot3D(x_tot, y_tot, z_tot, 'green',label='Total Stress')


    # Deviatoric stress
    x_dev = np.array([p, sig_main[0]])
    y_dev = np.array([p, sig_main[1]])
    z_dev = np.array([p,sig_main[2]])
    ax.plot3D(x_dev, y_dev, z_dev, 'blue',label='Deviatoric Stress')


    # Coordinate system
    axis_0 = np.linspace(0,0,2)
    axis_1 = np.linspace(0,np.max([p,0.1]),2)
    ax.plot3D(axis_1,axis_0,axis_0, 'black')
    ax.plot3D(axis_0,axis_1,axis_0, 'black')
    ax.plot3D(axis_0,axis_0,axis_1, 'black')


    # Set axis ratio equal for 3D plot
    xmin=x_tot.min()
    xmax=x_tot.max()
    ymin=y_tot.min()
    ymax=y_tot.max()
    zmin=z_tot.min()
    zmax=z_tot.max()

    max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)

    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')


    # Pi Plane
    point  = np.array([p,p,p])
    normal = point
    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d = -point.dot(normal)

    x_pi_plane, y_pi_plane = np.meshgrid(np.linspace(plt.xlim()[0], plt.xlim()[1], 100), np.linspace(plt.ylim()[0], plt.ylim()[1], 100))
    z_pi_plane = (-normal[0] * x_pi_plane - normal[1] * y_pi_plane - d) * 1. /normal[2]
 
    # Delete z-values outside plot window
    for i in np.arange(len(x_pi_plane)):
        for j in np.arange(len(x_pi_plane)):
	        if z_pi_plane[j,i] < Zb.min() or z_pi_plane[j,i] > Zb.max():
		        z_pi_plane[j,i] = np.nan
              
    ax.plot_surface(x_pi_plane,y_pi_plane,z_pi_plane,alpha=0.1,color='orange')
    ax.plot3D([0,0],[0,0],[0,0], 'orange',alpha=0.1,label='Pi Plane')


    # rotate the axes and update
    for angle in range(0, 360):
        ax.view_init(azim=0, elev=45)

    ax.set_xlabel('$\sigma_1$')
    ax.set_ylabel('$\sigma_2$')
    ax.set_zlabel('$\sigma_3$')

    plt.title('Principal Stress Space')

    plt.legend()
    plt.show()


def treloar_data_fit():
    if notebook_is_interactive:
        treloar_data_fit_interactive  = widgets.interactive(treloar_data_fit_eval,
            {'manual':True},
            Neo_Hooke=True,
            Mooney_Rivlin=False,
            Ogden_n_1=False,
            Ogden_n_2=False,
            Ogden_n_3=True)
        display(treloar_data_fit_interactive)
    else:
        treloar_data_fit_eval(True,False,False,False,True)
        
def treloar_data_fit_eval(Neo_Hooke,Mooney_Rivlin,Ogden_n_1,Ogden_n_2,Ogden_n_3):
    
    

    fig,axs=plt.subplots()
    

    lam_tension=np.array([1,1.01,1.12,1.24,1.39,1.61,1.89,2.17,2.42,3.01,3.58,4.03,4.76,5.36,5.76,6.16,6.4,6.62,6.87,7.05,7.16,7.27,7.43,7.5,7.61])
    stress_tension=np.array([0,0.03,0.14,0.23,0.32,0.41,0.5,0.58,0.67,0.85,1.04,1.21,1.58,1.94,2.29,2.67,3.02,3.39,3.75,4.12,4.47,4.85,5.21,5.57,6.3])

    plt.xlabel("$\lambda \;[\; ]$")
    plt.ylabel("$P_1 \; [MPa]$")
    plt.title('Uniaxial Tension')

    plt.plot(lam_tension,stress_tension,'--',marker='o')


    lam_tension_ges=np.linspace(lam_tension.min(),lam_tension.max(),100)

    if Neo_Hooke:
        stress_tension_Neo_Hooke=0.5250*(lam_tension_ges-lam_tension_ges**(-2.0))
        plt.plot(lam_tension_ges,stress_tension_Neo_Hooke,label='Neo-Hooke')

    if Mooney_Rivlin:
        stress_tension_Mooney_Rivlin=0.2659*(2.0*lam_tension_ges-2.0*lam_tension_ges**(-2.0))-0.0017*(2.0-2.0*lam_tension_ges**(-3.0))
        plt.plot(lam_tension_ges,stress_tension_Mooney_Rivlin,label='Mooney-Rivlin')
	
    if Ogden_n_1:
        stress_tension_Ogden_1=0.0980*(lam_tension_ges**(2.9473-1.0)-lam_tension_ges**(-0.5*2.9473-1.0))
        plt.plot(lam_tension_ges,stress_tension_Ogden_1,label='Ogden (n=1)')
	
    if Ogden_n_2:
        stress_tension_Ogden_2=0.3528*(lam_tension_ges**(2.05-1.0)-lam_tension_ges**(-0.5*2.05-1.0))+1.032e-9*(lam_tension_ges**(11.771-1.0)-lam_tension_ges**(-0.5*11.771-1.0))
        plt.plot(lam_tension_ges,stress_tension_Ogden_2,label='Ogden (n=2)')
	
    if Ogden_n_3:
        stress_tension_Ogden_3=0.0662*(lam_tension_ges**(2.875-1.0)-lam_tension_ges**(-0.5*2.875-1.0))+5.875e-12*(lam_tension_ges**(14.221-1.0)-lam_tension_ges**(-0.5*14.221-1.0))+0.6249*(lam_tension_ges**(1.0-1.0)-lam_tension_ges**(-0.5*1.0-1.0))
        plt.plot(lam_tension_ges,stress_tension_Ogden_3,label='Ogden (n=3)')
	
    plt.legend()
    plt.show()
    
    fig,axs=plt.subplots()

    lam_shear=np.array([1,1.06,1.14,1.21,1.32,1.46,1.87,2.4,2.98,3.48,3.96,4.36,4.69,4.96])
    stress_shear=np.array([0,0.07,0.16,0.24,0.33,0.42,0.59,0.76,0.93,1.11,1.28,1.46,1.62,1.79])

    plt.xlabel("$\lambda \;[\; ]$")
    plt.ylabel("$P_1 \; [MPa]$")
    plt.title('Pure Shear')

    plt.plot(lam_shear,stress_shear,'--',marker='o')


    lam_shear_ges=np.linspace(lam_shear.min(),lam_shear.max(),100)

    if Neo_Hooke:
        stress_shear_Neo_Hooke=0.5250*(lam_shear_ges-lam_shear_ges**(-3.0))
        plt.plot(lam_shear_ges,stress_shear_Neo_Hooke,label='Neo-Hooke')
	
    if Mooney_Rivlin:
        stress_shear_Mooney_Rivlin=0.2659*(2.0*lam_shear_ges-2.0*lam_shear_ges**(-3.0))-0.0017*(2.0*lam_shear_ges-2.0*lam_shear_ges**(-3.0))
        plt.plot(lam_shear_ges,stress_shear_Mooney_Rivlin,label='Mooney-Rivlin')

    if Ogden_n_1:
        stress_shear_Ogden_1=0.0980*(lam_shear_ges**(2.9473-1.0)-lam_shear_ges**(-2.9473-1.0))
        plt.plot(lam_shear_ges,stress_shear_Ogden_1,label='Ogden (n=1)')
	
    if Ogden_n_2:
        stress_shear_Ogden_2=0.3528*(lam_shear_ges**(2.05-1.0)-lam_shear_ges**(-2.05-1.0))+1.032e-9*(lam_shear_ges**(11.771-1.0)-lam_shear_ges**(-11.771-1.0))
        plt.plot(lam_shear_ges,stress_shear_Ogden_2,label='Ogden (n=2)')
	
    if Ogden_n_3:
        stress_shear_Ogden_3=0.0662*(lam_shear_ges**(2.875-1.0)-lam_shear_ges**(-2.875-1.0))+5.875e-12*(lam_shear_ges**(14.221-1.0)-lam_shear_ges**(-14.221-1.0))+0.6249*(lam_shear_ges**(1.0-1.0)-lam_shear_ges**(-1.0-1.0))
        plt.plot(lam_shear_ges,stress_shear_Ogden_3,label='Ogden (n=3)')

    plt.legend()
    plt.show()
    
    fig,axs=plt.subplots()

    lam_equi=np.array([1,1.04,1.08,1.12,1.14,1.2,1.31,1.42,1.69,1.94,2.49,3.03,3.43,3.75,4.03,4.26,4.44])
    stress_equi=np.array([0,0.09,0.16,0.24,0.26,0.33,0.44,0.51,0.65,0.77,0.96,1.24,1.45,1.72,1.96,2.22,2.43])

    plt.xlabel("$\lambda \;[\; ]$")
    plt.ylabel("$P_1 \; [MPa]$")
    plt.title('Equibiaxial Tension')

    plt.plot(lam_equi,stress_equi,'--',marker='o')


    lam_equi_ges=np.linspace(lam_equi.min(),lam_equi.max(),100)

    if Neo_Hooke:
        stress_equi_Neo_Hooke=0.5250*(lam_equi_ges-lam_equi_ges**(-5.0))
        plt.plot(lam_equi_ges,stress_equi_Neo_Hooke,label='Neo-Hooke')
	
	
    if Mooney_Rivlin:
        stress_equi_Mooney_Rivlin=0.2659*(2.0*lam_equi_ges-2.0*lam_equi_ges**(-5.0))-0.0017*(2.0*lam_equi_ges**(3.0)-2.0*lam_equi_ges**(-3.0))
        plt.plot(lam_equi_ges,stress_equi_Mooney_Rivlin,label='Mooney-Rivlin')
        
    if Ogden_n_1:
        stress_equi_Ogden_1=0.0980*(lam_equi_ges**(2.9473-1.0)-lam_equi_ges**(-2.0*2.9473-1.0))
        plt.plot(lam_equi_ges,stress_equi_Ogden_1,label='Ogden (n=1)')
	
    if Ogden_n_2:
        stress_equi_Ogden_2=0.3528*(lam_equi_ges**(2.05-1.0)-lam_equi_ges**(-2.0*2.05-1.0))+1.032e-9*(lam_equi_ges**(11.771-1.0)-lam_equi_ges**(-2.0*11.771-1.0))
        plt.plot(lam_equi_ges,stress_equi_Ogden_2,label='Ogden (n=2)')
	
    if Ogden_n_3:
        stress_equi_Ogden_3=0.0662*(lam_equi_ges**(2.875-1.0)-lam_equi_ges**(-2.0*2.875-1.0))+5.875e-12*(lam_equi_ges**(14.221-1.0)-lam_equi_ges**(-2.0*11.771-1.0))+0.6249*(lam_equi_ges**(1.0-1.0)-lam_equi_ges**(-2.0*1.0-1.0))
        plt.plot(lam_equi_ges,stress_equi_Ogden_3,label='Ogden (n=3)')

    plt.legend()
    plt.show()


def cubic_anisotropy_visualization():
    
    fig,axs=plt.subplots()
    ax = plt.axes(projection='3d')

    # Angles for spherical coordinates
    n_max=75
    phi, theta = np.meshgrid(np.linspace(0.0, 2.0*np.pi, n_max), np.linspace(0.0, 2.0*np.pi, n_max))

    # Normal vectors
    n=np.zeros([n_max,n_max,3])
    n[:,:,0]=np.cos(phi)*np.sin(theta)
    n[:,:,1]=np.sin(phi)*np.sin(theta)
    n[:,:,2]=np.cos(theta)

    # Basic tensors
    I2 = np.identity(3)
    IdI = np.tensordot(I2,I2,axes=0)
    I4 = IdI.transpose(0,2,1,3)
    IS = (I4 + I4.transpose(0,1,3,2))/2.0

    # Projectors of cubic materials
    Pc1 = IdI/3.0
    Dc = np.zeros([3,3,3,3])
    Dc[0,0,0,0] = 1.0
    Dc[1,1,1,1] = 1.0
    Dc[2,2,2,2] = 1.0
    Pc2 = Dc - Pc1
    Pc3 = IS - Pc1 - Pc2

    # Spectral representation of cubic compliance
    comp = Pc1/10.0 + Pc2 / 3.0 + Pc3 / 8.0

    # Initialize arrays
    x=np.zeros([n_max,n_max])
    y=np.zeros([n_max,n_max])
    z=np.zeros([n_max,n_max])
    r_ges=np.zeros([n_max,n_max])

    colors=np.empty((n_max,n_max), dtype=object)

    # Calculate stiffness 'r' in each direction
    for i in range(n_max):
        for j in range(n_max):
            n2 = np.tensordot(n[i,j,:],n[i,j,:],axes=0)
            n4 = np.tensordot(n2,n2,axes=0)
            mult=comp*n4
            r=1.0/mult.sum()
            x[i,j]=r*np.sin(theta[i,j])*np.cos(phi[i,j])
            y[i,j]=r*np.sin(theta[i,j])*np.sin(phi[i,j])
            z[i,j]=r*np.cos(theta[i,j])
            r_ges[i,j]=r
		
    r_min=r_ges.min()
    r_max=r_ges.max()

    # Calculate color dependent on 'r' in each direction
    for i in range(n_max):
        for j in range(n_max):
	        colors[i,j]=(1.0,(1.0-(r_ges[i,j]-r_min)/(r_max-r_min))*1.0,0.0)

    ax.plot_surface(x,y,z, shade=False,
                       linewidth=0.5,facecolors=colors)
				   
				   
    # Set axis ratio equal for 3D plot
    xmin=x.min()
    xmax=x.max()
    ymin=y.min()
    ymax=y.max()
    zmin=z.min()
    zmax=z.max()

    max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)

    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')

    # rotate the axes and update
    for angle in range(0, 360):
        ax.view_init(azim=45, elev=45)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')

    plt.title("Young's modulus for cubic anisotropy")
    plt.show()
    
# Initialize notebook with static content    
display(Javascript("Jupyter.notebook.execute_cells([5])"))