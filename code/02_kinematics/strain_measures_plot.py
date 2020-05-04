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


