
lam_tension=np.array([1,1.01,1.12,1.24,1.39,1.61,1.89,2.17,2.42,3.01,3.58,4.03,4.76,5.36,5.76,6.16,6.4,6.62,6.87,7.05,7.16,7.27,7.43,7.5,7.61])
stress_tension=np.array([0,0.03,0.14,0.23,0.32,0.41,0.5,0.58,0.67,0.85,1.04,1.21,1.58,1.94,2.29,2.67,3.02,3.39,3.75,4.12,4.47,4.85,5.21,5.57,6.3])

plt.xlabel("$\lambda \;[\; ]$")
plt.ylabel("$P_1 \; [MPa]$")
plt.title('Uniaxial Tension')

plt.plot(lam_tension,stress_tension,'--',marker='o')


lam_tension_ges=np.linspace(lam_tension.min(),lam_tension.max(),100)

if w_Neo_Hooke.value:
    stress_tension_Neo_Hooke=0.5250*(lam_tension_ges-lam_tension_ges**(-2.0))
    plt.plot(lam_tension_ges,stress_tension_Neo_Hooke,label='Neo-Hooke')

	
if w_Ogden_1.value:
    stress_tension_Ogden_1=0.0980*(lam_tension_ges**(2.9473-1.0)-lam_tension_ges**(-0.5*2.9473-1.0))
    plt.plot(lam_tension_ges,stress_tension_Ogden_1,label='Ogden (n=1)')
	
if w_Ogden_2.value:
    stress_tension_Ogden_2=0.3528*(lam_tension_ges**(2.05-1.0)-lam_tension_ges**(-0.5*2.05-1.0))+1.032e-9*(lam_tension_ges**(11.771-1.0)-lam_tension_ges**(-0.5*11.771-1.0))
    plt.plot(lam_tension_ges,stress_tension_Ogden_2,label='Ogden (n=2)')
	
if w_Ogden_3.value:
    stress_tension_Ogden_3=0.0662*(lam_tension_ges**(2.875-1.0)-lam_tension_ges**(-0.5*2.875-1.0))+5.875e-12*(lam_tension_ges**(14.221-1.0)-lam_tension_ges**(-0.5*14.221-1.0))+0.6249*(lam_tension_ges**(1.0-1.0)-lam_tension_ges**(-0.5*1.0-1.0))
    plt.plot(lam_tension_ges,stress_tension_Ogden_3,label='Ogden (n=3)')
	
if w_Mooney_Rivlin.value:
    stress_tension_Mooney_Rivlin=0.2659*(2.0*lam_tension_ges-2.0*lam_tension_ges**(-2.0))-0.0017*(2.0-2.0*lam_tension_ges**(-3.0))
    plt.plot(lam_tension_ges,stress_tension_Mooney_Rivlin,label='Mooney-Rivlin')

plt.legend()
plt.show()


lam_shear=np.array([1,1.06,1.14,1.21,1.32,1.46,1.87,2.4,2.98,3.48,3.96,4.36,4.69,4.96])
stress_shear=np.array([0,0.07,0.16,0.24,0.33,0.42,0.59,0.76,0.93,1.11,1.28,1.46,1.62,1.79])

plt.xlabel("$\lambda \;[\; ]$")
plt.ylabel("$P_1 \; [MPa]$")
plt.title('Pure Shear')

plt.plot(lam_shear,stress_shear,'--',marker='o')


lam_shear_ges=np.linspace(lam_shear.min(),lam_shear.max(),100)

if w_Neo_Hooke.value:
    stress_shear_Neo_Hooke=0.5250*(lam_shear_ges-lam_shear_ges**(-3.0))
    plt.plot(lam_shear_ges,stress_shear_Neo_Hooke,label='Neo-Hooke')
	
if w_Ogden_1.value:
    stress_shear_Ogden_1=0.0980*(lam_shear_ges**(2.9473-1.0)-lam_shear_ges**(-2.9473-1.0))
    plt.plot(lam_shear_ges,stress_shear_Ogden_1,label='Ogden (n=1)')
	
if w_Ogden_2.value:
    stress_shear_Ogden_2=0.3528*(lam_shear_ges**(2.05-1.0)-lam_shear_ges**(-2.05-1.0))+1.032e-9*(lam_shear_ges**(11.771-1.0)-lam_shear_ges**(-11.771-1.0))
    plt.plot(lam_shear_ges,stress_shear_Ogden_2,label='Ogden (n=2)')
	
if w_Ogden_3.value:
    stress_shear_Ogden_3=0.0662*(lam_shear_ges**(2.875-1.0)-lam_shear_ges**(-2.875-1.0))+5.875e-12*(lam_shear_ges**(14.221-1.0)-lam_shear_ges**(-14.221-1.0))+0.6249*(lam_shear_ges**(1.0-1.0)-lam_shear_ges**(-1.0-1.0))
    plt.plot(lam_shear_ges,stress_shear_Ogden_3,label='Ogden (n=3)')
	
if w_Mooney_Rivlin.value:
    stress_shear_Mooney_Rivlin=0.2659*(2.0*lam_shear_ges-2.0*lam_shear_ges**(-3.0))-0.0017*(2.0*lam_shear_ges-2.0*lam_shear_ges**(-3.0))
    plt.plot(lam_shear_ges,stress_shear_Mooney_Rivlin,label='Mooney-Rivlin')

plt.legend()
plt.show()


lam_equi=np.array([1,1.04,1.08,1.12,1.14,1.2,1.31,1.42,1.69,1.94,2.49,3.03,3.43,3.75,4.03,4.26,4.44])
stress_equi=np.array([0,0.09,0.16,0.24,0.26,0.33,0.44,0.51,0.65,0.77,0.96,1.24,1.45,1.72,1.96,2.22,2.43])

plt.xlabel("$\lambda \;[\; ]$")
plt.ylabel("$P_1 \; [MPa]$")
plt.title('Equibiaxial Tension')

plt.plot(lam_equi,stress_equi,'--',marker='o')


lam_equi_ges=np.linspace(lam_equi.min(),lam_equi.max(),100)

if w_Neo_Hooke.value:
    stress_equi_Neo_Hooke=0.5250*(lam_equi_ges-lam_equi_ges**(-5.0))
    plt.plot(lam_equi_ges,stress_equi_Neo_Hooke,label='Neo-Hooke')
	
if w_Ogden_1.value:
    stress_equi_Ogden_1=0.0980*(lam_equi_ges**(2.9473-1.0)-lam_equi_ges**(-2.0*2.9473-1.0))
    plt.plot(lam_equi_ges,stress_equi_Ogden_1,label='Ogden (n=1)')
	
if w_Ogden_2.value:
    stress_equi_Ogden_2=0.3528*(lam_equi_ges**(2.05-1.0)-lam_equi_ges**(-2.0*2.05-1.0))+1.032e-9*(lam_equi_ges**(11.771-1.0)-lam_equi_ges**(-2.0*11.771-1.0))
    plt.plot(lam_equi_ges,stress_equi_Ogden_2,label='Ogden (n=2)')
	
if w_Ogden_3.value:
    stress_equi_Ogden_3=0.0662*(lam_equi_ges**(2.875-1.0)-lam_equi_ges**(-2.0*2.875-1.0))+5.875e-12*(lam_equi_ges**(14.221-1.0)-lam_equi_ges**(-2.0*11.771-1.0))+0.6249*(lam_equi_ges**(1.0-1.0)-lam_equi_ges**(-2.0*1.0-1.0))
    plt.plot(lam_equi_ges,stress_equi_Ogden_3,label='Ogden (n=3)')
	
if w_Mooney_Rivlin.value:
    stress_equi_Mooney_Rivlin=0.2659*(2.0*lam_equi_ges-2.0*lam_equi_ges**(-5.0))-0.0017*(2.0*lam_equi_ges**(3.0)-2.0*lam_equi_ges**(-3.0))
    plt.plot(lam_equi_ges,stress_equi_Mooney_Rivlin,label='Mooney-Rivlin')

plt.legend()
plt.show()