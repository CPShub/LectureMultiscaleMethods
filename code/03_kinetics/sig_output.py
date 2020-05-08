sig_inp=list(w_sig.result.values())
sigma=np.eye(3)
sigma[0,0]=sig_inp[0]
sigma[1,1]=sig_inp[1]
sigma[2,2]=sig_inp[2]
sigma[0,1]=sig_inp[3]
sigma[0,2]=sig_inp[4]
sigma[1,2]=sig_inp[5]

sigma[1,0]=sigma[0,1]
sigma[2,0]=sigma[0,2]
sigma[2,1]=sigma[1,2]

p=np.trace(sigma)/3.0

sigma_dev=sigma-p*np.identity(3)

print('Cauchy Stress Tensor')
print()
print('sigma =')
print(sigma)
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