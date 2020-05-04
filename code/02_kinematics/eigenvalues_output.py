F_inp=list(w_F.result.values())
F=np.eye(3)
F[0,0]=F_inp[0]
F[1,1]=F_inp[1]
F[1,0]=F_inp[2]
F[0,1]=F_inp[3]


#	Eigenvalues
lam_U, eigv_U = np.linalg.eig(U)

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
