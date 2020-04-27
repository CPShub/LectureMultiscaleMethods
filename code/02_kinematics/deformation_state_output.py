#lam_quad, eigv_E = np.linalg.eig(b)

F=np.eye(3)
F[0,0]=F_inp[0]
F[1,1]=F_inp[1]
F[1,0]=F_inp[2]
F[0,1]=F_inp[3]
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