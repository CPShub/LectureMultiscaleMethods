F_inp=list(w_F.result.values())
F=np.eye(3)
F[0,0]=F_inp[0]
F[1,1]=F_inp[1]
F[1,0]=F_inp[2]
F[0,1]=F_inp[3]

#	Polar Decomposition
u,v=sp.linalg.polar(F, side='left')

#	Deviatoric / Volumetric Decomposition
F_r=np.matmul(u,np.eye(3))
F_rd=np.matmul(v,F_r)/np.linalg.det(v)**(1/3)
F_rdv=F_rd*np.linalg.det(v)**(1/3)

