F_inp=list(w_F.result.values())
F=np.eye(3)
F[0,0]=F_inp[0]
F[1,1]=F_inp[1]
F[1,0]=F_inp[2]
F[0,1]=F_inp[3]


#	Eigenvalues
lam_E, eigv_E = np.linalg.eig(E)
lam_e, eigv_e = np.linalg.eig(e)

fig,axs=plt.subplots()

plt.plot([0,1,1,0,0],[0,0,1,1,0])
plt.plot([0,F_rdv[0,0],F_rdv[0,0]+F_rdv[0,1],F_rdv[0,1],0],[0,F_rdv[1,0],F_rdv[1,0]+F_rdv[1,1],F_rdv[1,1],0], linewidth=2,linestyle='--')

plt.arrow(0, 0, lam_E[0]*eigv_E[0,0], lam_E[0]*eigv_E[1,0], length_includes_head=True,
          head_width=0.1, head_length=0.1,color='red')
plt.arrow(0, 0, lam_E[1]*eigv_E[0,1], lam_E[1]*eigv_E[1,1], length_includes_head=True,
          head_width=0.1, head_length=0.1,color='red')
		  
GL=plt.plot(0, 0, lam_E[0]*eigv_E[0,0], lam_E[0]*eigv_E[1,0],color='red')
plt.plot(0, 0, lam_E[1]*eigv_E[0,1], lam_E[1]*eigv_E[1,1],color='red')

plt.arrow(0, 0, lam_e[0]*eigv_e[0,0], lam_e[0]*eigv_e[1,0], length_includes_head=True,
          head_width=0.1, head_length=0.1,color='green')
plt.arrow(0, 0, lam_e[1]*eigv_e[0,1], lam_e[1]*eigv_e[1,1], length_includes_head=True,
          head_width=0.1, head_length=0.1,color='green')
		  
EA=plt.plot(0, 0, lam_e[0]*eigv_e[0,0], lam_e[0]*eigv_e[1,0],color='green')
plt.plot(0, 0, lam_e[1]*eigv_e[0,1], lam_e[1]*eigv_e[1,1],color='green')
		  
ax = plt.gca()
plt.gca().legend(('Green-Lagrange','Euler-Almansi'))
leg = ax.get_legend()
leg.legendHandles[0].set_color('red')
leg.legendHandles[1].set_linestyle('-')
leg.legendHandles[1].set_color('green')
leg.legendHandles[1].set_linestyle('-')


axs.axis('equal')
plt.show()

print('Eigenvalues Green-Lagrange')
print(lam_E)
print('Eigenvalues Euler-Almansi')
print(lam_e)