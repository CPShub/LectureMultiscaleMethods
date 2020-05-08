F_inp=list(w_F.result.values())
F=np.eye(3)
F[0,0]=F_inp[0]
F[1,1]=F_inp[1]
F[0,1]=F_inp[2]
F[1,0]=F_inp[3]

F[2,2]=1.0/(F[0,0]*F[1,1])


#	Strain Tensors
b=np.matmul(F,np.transpose(F))
E=0.5*(np.matmul(np.transpose(F),F)-np.identity(3))
e=0.5*(np.identity(3)-np.linalg.inv(b))
e_lin=0.5*(F+np.transpose(F))-np.identity(3)

J=np.linalg.det(F)


#	Stress Tensors
#	Parameter
mu=12
lam=8

S=2.0*mu*E+lam*np.trace(E)*np.identity(3)
P=np.matmul(F,S)
sigma=1.0/J*np.matmul(P,np.transpose(F))


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
#print()
#print('J =')
#print(np.linalg.det(F))