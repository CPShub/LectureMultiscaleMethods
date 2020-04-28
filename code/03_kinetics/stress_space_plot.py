
fig,axs=plt.subplots()
#fig = plt.figure()
ax = plt.axes(projection='3d')
ax = plt.axes(projection='3d')

# Hydrostatic axis
p_star=p*0.577
z_p_axis = np.linspace(-p_star*1.1, p_star*1.1, 2)
x_p_axis = np.linspace(-p_star*1.1, p_star*1.1, 2)
y_p_axis = np.linspace(-p_star*1.1, p_star*1.1, 2)
ax.plot3D(x_p_axis , y_p_axis , z_p_axis , 'black',linestyle='--')

# Hydrostatic stress
z_p = np.linspace(0, p_star, 2)
x_p = np.linspace(0, p_star, 2)
y_p = np.linspace(0, p_star, 2)
ax.plot3D(x_p, y_p, z_p, 'red')

# Deviatoric stress
sig_pa, v = np.linalg.eig(sigma_dev)

x_dev = [0, sig_pa[0]] +p_star
y_dev = [0, sig_pa[1]] +p_star
z_dev = [0,sig_pa[2]]  +p_star

ax.plot3D(x_dev, y_dev, z_dev, 'green')


axis_0 = np.linspace(0,0,2)
axis_1 = np.linspace(0,p_star,2)

ax.plot3D(axis_1,axis_0,axis_0, 'black')
ax.plot3D(axis_0,axis_1,axis_0, 'black')
ax.plot3D(axis_0,axis_0,axis_1, 'black')


# rotate the axes and update
for angle in range(0, 360):
   ax.view_init(90, 90)
axs.axis('equal')
plt.show()