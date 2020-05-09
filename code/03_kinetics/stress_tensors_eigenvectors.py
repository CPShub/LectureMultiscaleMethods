# Eigenvalues
lam_sig, eigv_sig = np.linalg.eig(sigma)
lam_S, eigv_S = np.linalg.eig(S)



fig,axs=plt.subplots()


plt.plot([0,1,1,0,0],[0,0,1,1,0])

F_inp=list(w_F.result.values())
F=np.eye(3)
F[0,0]=F_inp[0]
F[1,1]=F_inp[1]
F[1,0]=F_inp[2]
F[0,1]=F_inp[3]
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
#plt.legend()
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