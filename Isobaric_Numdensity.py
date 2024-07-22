def cal_numd_P(T_array, P0, T0):
    from exojax.atm.idealgas import number_density
    from exojax.utils.constants import kB 
    
    
    #Calculate the total number density and volume at the initial condition by PV=n*k_b*T
    n_total = number_density(P0,T0) #P[bar]. n=P/k_b*T
    #V_total = (n_total * kB * T0) / P0
    nlayer = len(T_array) #number of Temperature layer
    
    '''
    #Calculate the number density at each region 
    Assumption(if there is "i" Temperature regions with same volume)
       V1 = V2 =...= Vi = V_total / i
       P1 = P2 =...= Pi = P_total 
       ∴ n1 * T1 = n2 * T2 = .... ni * Ti, 
         ni = T1 * n1 / Ti
    
    Also, the Total amount of molecule is equal to the sum of each amount of molecule 
        n_total * V_total = n1 * V1 +...+ ni * Vi
        n_total * i * V1 = (n1 + T1 / T2 * n1 + T1 / T3 * n1 +...+ T1 / Ti * n1) * V1 
        n_total * i = n1 * T1 * (1 / T1 + 1 / T2 +...+1 / Ti) 
        ∴ n1 = n_total * i / (T1 * Tinv_sum), (Tinv_sum = 1 / T1 + 1 / T2 +...+1 / Ti)

        P_total = P1 = n1 * k_b * T1
                     = n_total * i /(T1 * Tinv_sum) * k_b * T1
                     = n_total * i * k_B /Tinv_sum 
        
    '''
    Tinv_sum = 0
    for i in range(nlayer):
        Tinv_sum += 1/T_array[i]
    n_array = n_total * nlayer / (T_array*Tinv_sum)

    #Calculate the pressure at given Temperatures(1e-6:conversion factor of the Pa to bar)
    P_total = nlayer * kB * 1.0e-6 * n_total/(Tinv_sum)
 
    #print(nlayer/Tinv_sum)
    return n_array,P_total

'''
#sample codes
import numpy as np
T_array = np.array([300,400,500])  # Array of temperatures (K)
P0 = 0.423  # Pressure at the initial condition (bar)
T0 = 296  # Temperature at the initial condition (K)

number_density, pressure = cal_numd_P(T_array, P0, T0)
print("Number Density:", number_density)
print("Pressure:", pressure)
'''
