n_max=50

El=np.zeros([3,3,n_max])
E_lin_l=np.zeros([3,3,n_max])
I1=np.zeros(n_max)

F_inp=list(w_F.result.values())

Fl_0_0=np.linspace(1,F_inp[0],n_max)
Fl_1_1=np.linspace(1,F_inp[1],n_max)
Fl_0_1=np.linspace(0,F_inp[2],n_max)
Fl_1_0=np.linspace(0,F_inp[3],n_max)

for i in list(range(0,n_max,1)):
    Fl=np.eye(3)
    Fl[0,0]=Fl_0_0[i]
    Fl[1,1]=Fl_1_1[i]
    Fl[0,1]=Fl_0_1[i]
    Fl[1,0]=Fl_1_0[i]
    #	Strain Tensors
    bl=np.matmul(Fl,np.transpose(Fl))
    El[:,:,i]=0.5*(np.matmul(np.transpose(Fl),Fl)-np.identity(3))
    E_lin_l[:,:,i]=0.5*(Fl+np.transpose(Fl))-np.identity(3)
    I1[i]=np.trace(bl)-3.0
	

plt.xlabel("$I_1(C)$")
plt.ylabel("11-Component")
plt.title("Green-Lagrange Strain Tensor and its Linearization")
plt.plot(I1,El[0,0,:],I1,E_lin_l[0,0,:])
plt.gca().legend(('Green-Lagrange Strain Tensor','Linearized Strain Tensor'))
plt.show()

plt.xlabel("$I_1(C)$")
plt.ylabel("12-Component")
plt.plot(I1,El[0,1,:],I1,E_lin_l[0,1,:])
plt.gca().legend(('Green-Lagrange Strain Tensor','Linearized Strain Tensor'))
plt.show()


