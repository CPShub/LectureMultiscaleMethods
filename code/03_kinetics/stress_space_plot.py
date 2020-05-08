fig,axs=plt.subplots()
ax = plt.axes(projection='3d')


# Hydrostatic axis
#p_axis = np.array([-0.1*p, p*1.1])
#ax.plot3D(p_axis , p_axis , p_axis , 'black',linestyle='--',label='Hydrostatic Axis')


# Hydrostatic stress
p_stress = np.array([0, p])
ax.plot3D(p_stress, p_stress, p_stress, 'red',label='Hydrostatic Stress')


# Total stress
sig_main, v = np.linalg.eig(sigma)
x_tot = np.array([0, sig_main[0]])
y_tot = np.array([0, sig_main[1]]) 
z_tot = np.array([0,sig_main[2]])  
ax.plot3D(x_tot, y_tot, z_tot, 'green',label='Total Stress')


# Deviatoric stress
sig_dev_main, v = np.linalg.eig(sigma_dev)
x_dev = np.array([p, sig_dev_main[0]+p])
y_dev = np.array([p, sig_dev_main[1]+p])
z_dev = np.array([p,sig_dev_main[2]+p])
ax.plot3D(x_dev, y_dev, z_dev, 'blue',label='Deviatoric Stress')


# Coordinate system
axis_0 = np.linspace(0,0,2)
axis_1 = np.linspace(0,np.max([p,0.1]),2)
ax.plot3D(axis_1,axis_0,axis_0, 'black')
ax.plot3D(axis_0,axis_1,axis_0, 'black')
ax.plot3D(axis_0,axis_0,axis_1, 'black')


# Set axis ratio equal for 3D plot
xmin=x_tot.min()
xmax=x_tot.max()
ymin=y_tot.min()
ymax=y_tot.max()
zmin=z_tot.min()
zmax=z_tot.max()

max_range = np.array([xmax-xmin, ymax-ymin, zmax-zmin]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)

for xb, yb, zb in zip(Xb, Yb, Zb):
   ax.plot([xb], [yb], [zb], 'w')


# Pi Plane
point  = np.array([p,p,p])
normal = point
# a plane is a*x+b*y+c*z+d=0
# [a,b,c] is the normal. Thus, we have to calculate
# d and we're set
d = -point.dot(normal)

x_pi_plane, y_pi_plane = np.meshgrid(np.linspace(plt.xlim()[0], plt.xlim()[1], 100), np.linspace(plt.ylim()[0], plt.ylim()[1], 100))
z_pi_plane = (-normal[0] * x_pi_plane - normal[1] * y_pi_plane - d) * 1. /normal[2]
 
# Delete z-values outside plot window
for i in np.arange(len(x_pi_plane)):
    for j in np.arange(len(x_pi_plane)):
	    if z_pi_plane[j,i] < Zb.min() or z_pi_plane[j,i] > Zb.max():
		    z_pi_plane[j,i] = np.nan
              
#  supress warning "NaN-Values" at surface plot
warnings.filterwarnings("ignore") 

ax.plot_surface(x_pi_plane,y_pi_plane,z_pi_plane,alpha=0.1,color='orange')
ax.plot3D([0,0],[0,0],[0,0], 'orange',alpha=0.1,label='Pi Plane')


# rotate the axes and update
for angle in range(0, 360):
    ax.view_init(azim=0, elev=45)

ax.set_xlabel('$\sigma_1$')
ax.set_ylabel('$\sigma_2$')
ax.set_zlabel('$\sigma_3$')

plt.legend()
plt.show()