import numpy as np


def cal_numd_P(T_array, P0, T0):
    from exojax.atm.idealgas import number_density
    from exojax.utils.constants import kB 
    
    # Calculate the total number density and volume at the initial condition
    n_total = number_density(P0,T0) #P[bar]
    V_total = (n_total * kB * T0) / P0
    nlayer = len(T_array)
    
    #Calculate the number density at each region    
    Tinv_sum = 0
    for i in range(nlayer):
        Tinv_sum += 1/T_array[i]
    n = nlayer*n_total/(T_array*Tinv_sum)

    #Calculate the pressure at given Temperatures by P = n*kb*T/V
    P_T = nlayer * kB * n_total/(V_total*Tinv_sum)
 
    #print(nlayer/Tinv_sum)
    return n,P_T


'''
#sample codes
T_array = np.array([300,400,500])  # Array of temperatures (K)
P0 = 0.423  # Pressure at the initial condition (bar)
T0 = 296  # Temperature at the initial condition (K)

number_density, pressure = cal_numd_P(T_array, P0, T0)
print("Number Density:", number_density)
print("Pressure:", pressure)
'''