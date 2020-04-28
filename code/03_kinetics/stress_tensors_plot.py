n_max=50

#	Parameter
mu=12
lam=8

Sl=np.zeros([3,3,n_max])
Pl=np.zeros([3,3,n_max])
sigmal=np.zeros([3,3,n_max])
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
    El=0.5*(np.matmul(np.transpose(Fl),Fl)-np.identity(3))
    el=0.5*(np.identity(3)-np.linalg.inv(bl))
    J=np.linalg.det(Fl)
    #	Stress Tensors
    Sl[:,:,i]=2.0*mu*El+lam*np.trace(El)*np.identity(3)
    Pl[:,:,i]=np.matmul(Fl,Sl[:,:,i])
    sigmal[:,:,i]=1.0/J*np.matmul(Pl[:,:,i],np.transpose(Fl))
    I1[i]=np.trace(bl)-3.0
	
	

plt.xlabel("$I_1(b)$")
plt.ylabel("11-Component in MPa")
plt.title("Comparison of Stress Measures")
plt.plot(I1,sigmal[0,0,:],I1,Sl[0,0,:],I1,Pl[0,0,:])
plt.gca().legend(('Cauchy','First Piola-Kirchhoff','Second Piola-Kirchhoff'))
plt.show()

plt.xlabel("$I_1(b)$")
plt.ylabel("12-Component in MPa")
plt.plot(I1,sigmal[0,1,:],I1,Sl[0,1,:],I1,Pl[0,1,:])
plt.gca().legend(('Cauchy','First Piola-Kirchhoff','Second Piola-Kirchhoff'))
plt.show()

