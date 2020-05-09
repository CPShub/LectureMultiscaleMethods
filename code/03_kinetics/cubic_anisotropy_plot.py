fig,axs=plt.subplots()
ax = plt.axes(projection='3d')

# Angles for spherical coordinates
n_max=75
phi, theta = np.meshgrid(np.linspace(0.0, 2.0*np.pi, n_max), np.linspace(0.0, 2.0*np.pi, n_max))

# Normal vectors
n=np.zeros([n_max,n_max,3])
n[:,:,0]=np.cos(phi)*np.sin(theta)
n[:,:,1]=np.sin(phi)*np.sin(theta)
n[:,:,2]=np.cos(theta)

# Basic tensors
I2 = np.identity(3)
IdI = np.tensordot(I2,I2,axes=0)
I4 = IdI.transpose(0,2,1,3)
IS = (I4 + I4.transpose(0,1,3,2))/2.0

# Projectors of cubic materials
Pc1 = IdI/3.0
Dc = np.zeros([3,3,3,3])
Dc[0,0,0,0] = 1.0
Dc[1,1,1,1] = 1.0
Dc[2,2,2,2] = 1.0
Pc2 = Dc - Pc1
Pc3 = IS - Pc1 - Pc2

# Spectral representation of cubic compliance
comp = Pc1/10.0 + Pc2 / 3.0 + Pc3 / 8.0

# Initialize arrays
x=np.zeros([n_max,n_max])
y=np.zeros([n_max,n_max])
z=np.zeros([n_max,n_max])
r_ges=np.zeros([n_max,n_max])

colors=np.empty((n_max,n_max), dtype=object)

# Calculate stiffness 'r' in each direction
for i in range(n_max):
    for j in range(n_max):
        n2 = np.tensordot(n[i,j,:],n[i,j,:],axes=0)
        n4 = np.tensordot(n2,n2,axes=0)
        mult=comp*n4
        r=1.0/mult.sum()
        x[i,j]=r*np.sin(theta[i,j])*np.cos(phi[i,j])
        y[i,j]=r*np.sin(theta[i,j])*np.sin(phi[i,j])
        z[i,j]=r*np.cos(theta[i,j])
        r_ges[i,j]=r
		
r_min=r_ges.min()
r_max=r_ges.max()

# Calculate color dependent on 'r' in each direction
for i in range(n_max):
    for j in range(n_max):
	    colors[i,j]=(1.0,(1.0-(r_ges[i,j]-r_min)/(r_max-r_min))*1.0,0.0)

ax.plot_surface(x,y,z, shade=False,
                   linewidth=0.5,facecolors=colors)
				   
				   
# Set axis ratio equal for 3D plot
xmin=x.min()
xmax=x.max()
ymin=y.min()
ymax=y.max()
zmin=z.min()
zmax=z.max()

max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)

for xb, yb, zb in zip(Xb, Yb, Zb):
   ax.plot([xb], [yb], [zb], 'w')

# rotate the axes and update
for angle in range(0, 360):
    ax.view_init(azim=0, elev=45)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')

plt.title("Young's modulus for cubic anisotropy")
plt.show()