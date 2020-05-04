F_inp=list(w_F.result.values())
F=np.eye(3)
F[0,0]=F_inp[0]
F[1,1]=F_inp[1]
F[1,0]=F_inp[2]
F[0,1]=F_inp[3]


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


#print('Euler-Almansi Strain Tensor')
#print(e)