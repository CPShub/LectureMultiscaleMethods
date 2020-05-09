
lam_tension=np.array([1,1.01,1.12,1.24,1.39,1.61,1.89,2.17,2.42,3.01,3.58,4.03,4.76,5.36,5.76,6.16,6.4,6.62,6.87,7.05,7.16,7.27,7.43,7.5,7.61])
stress_tension_P=np.array([0,0.03,0.14,0.23,0.32,0.41,0.5,0.58,0.67,0.85,1.04,1.21,1.58,1.94,2.29,2.67,3.02,3.39,3.75,4.12,4.47,4.85,5.21,5.57,6.3])
stress_tension_S=stress_tension_P/lam_tension
stress_tension_sigma=stress_tension_P*lam_tension

plt.xlabel("$\lambda \;[\; ]$")
plt.ylabel("$11-component \; [MPa]$")
plt.title('Uniaxial Tension')

plt.plot(lam_tension,stress_tension_S,'--',marker='o',label='S')
plt.plot(lam_tension,stress_tension_P,'--',marker='o',label='P')
plt.plot(lam_tension,stress_tension_sigma,'--',marker='o',label='sigma')

plt.legend()
plt.show()


lam_shear=np.array([1,1.06,1.14,1.21,1.32,1.46,1.87,2.4,2.98,3.48,3.96,4.36,4.69,4.96])
stress_shear_P=np.array([0,0.07,0.16,0.24,0.33,0.42,0.59,0.76,0.93,1.11,1.28,1.46,1.62,1.79])
stress_shear_S=stress_shear_P/lam_shear
stress_shear_sigma=stress_shear_P*lam_shear

plt.xlabel("$\lambda \;[\; ]$")
plt.ylabel("$11-component \; [MPa]$")
plt.title('Pure Shear')

plt.plot(lam_shear,stress_shear_S,'--',marker='o',label='S')
plt.plot(lam_shear,stress_shear_P,'--',marker='o',label='P')
plt.plot(lam_shear,stress_shear_sigma,'--',marker='o',label='sigma')

plt.legend()
plt.show()
