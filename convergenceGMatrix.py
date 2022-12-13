import numpy as np
import matplotlib.pyplot as plt
from loadMatrix import loadMatrix

plt.rcParams['text.usetex'] = True

## To Be adaptated
datedir = ["10_12_2022", "11_12_2022"]
datefile = ["101222", "111222"]
matrixClass = ["G01", "G02", "G03", "G11", "G12", "G2"]
epslist = ["064", "032", "016", "008", "004", "002", "001"]
epsilon = np.array([0.64, 0.32, 0.16, 0.08, 0.04, 0.02, 0.01])

# Parameter
pi=np.pi
V_star = -1./(8*pi*pi)
N_eps = len(epslist)
N_date = len(datedir)
P = 10 #P should be in inferior to nev define in LaplacianEigenvalue.edp
error = np.zeros((N_eps, 6, N_date))

######################
### Precomputation ###
######################
# Compute eigenvalue of Laplacian recquired to compute limit value
def LaplacianEigenvalue() :
    eigenvalue = np.zeros(P*P)
    compt=0
    for i in range(1, P+1) :
        for j in range(1, P+1) :
            eigenvalue[compt] = (i*i + j*j)*pi*pi
            compt+=1
    eigenvalue = np.sort(eigenvalue)
    return(eigenvalue)


#Compute converged matrix
def ConvergedMatrix() :
    eigenvalue = LaplacianEigenvalue()
    ## G01_lim
    G01 = np.zeros((P,P))
    for i in range(P) : 
        G01[i,i] = 1./(eigenvalue[i]+V_star)**2
    ## G02_lim
    G02 = np.zeros((P,P))
    for i in range(P) : 
        G02[i,i] = 1./((eigenvalue[i]+V_star)*eigenvalue[i])
    ## G03_lim
    G03 = np.zeros((P,P))
    for i in range(P) : 
        G03[i,i] = 1./(eigenvalue[i]*eigenvalue[i])
    ## G11_lim
    G11 = np.zeros((P,P))
    for i in range(P) : 
        G11[i,i] = 1./(eigenvalue[i]*(eigenvalue[i]+V_star)**2)
    ## G12_lim
    G12 = np.zeros((P,P))
    for i in range(P) : 
        G12[i,i] = 1./((eigenvalue[i]+V_star)*eigenvalue[i]**2)
    ## G12_lim
    G2 = np.zeros((P,P))
    for i in range(P) : 
        G2[i,i] = 1./((eigenvalue[i]+V_star)*eigenvalue[i])**2
    
    return([G01, G02, G03, G11, G12, G2])
    
convergedmatrix = ConvergedMatrix()



print(N_date)

for d in range(N_date) :
    for g, G in enumerate(matrixClass):
        # For each class of matrix G, we compute the L2 error for each epsilon
        for e, eps in enumerate(epslist) :
            filename = "./results/"+str(datedir[d])+"/Precomputation_"+str(G)+"_eps"+str(eps)+"_"+str(datefile[d])+".txt"
            GMatrix = loadMatrix(filename)
            G_star = convergedmatrix[g]
            error[e, g, d] = np.linalg.norm(GMatrix-G_star)/np.linalg.norm(G_star)

print(error)

plt.plot(epsilon, error[:, 0, 0], color='red', marker='+', label='G01 $= \langle u_\epsilon(f_i),u_\epsilon(f_j)$' + r'$\rangle$' + '$ h=0.001$')
plt.plot(epsilon, error[:, 0, 1], color='red', marker='+', label='G01 $= \langle u_\epsilon(f_i),u_\epsilon(f_j)$' + r'$\rangle$' + '$ h=0.005$')
plt.plot(epsilon, error[:, 1, 0], color='blue', marker='+', label='G02 $= \langle u_\epsilon(f_i),(-\Delta)^{-1}(f_j)$' + r'$\rangle$' + '$ h=0.001$')
plt.plot(epsilon, error[:, 1, 1], color='blue', marker='+', label='G02 $= \langle u_\epsilon(f_i),(-\Delta)^{-1}(f_j)$' + r'$\rangle$' + '$ h=0.005$')
#plt.plot(epsilon, error[:, 2, 0], color='green', marker='+', label='G03 $= \langle (-\Delta)^{-1}(f_i),(-\Delta)^{-1}(f_j)$' + r'$\rangle$')
plt.plot(epsilon, error[:, 3, 0], color='purple', marker='+', label='G11 $= \langle (-\Delta)^{-1}u_\epsilon(f_i),u_\epsilon(f_j)$' + r'$\rangle$' + '$ h=0.001$')
plt.plot(epsilon, error[:, 3, 1], color='purple', marker='+', label='G11 $= \langle (-\Delta)^{-1}u_\epsilon(f_i),u_\epsilon(f_j)$' + r'$\rangle$' + '$ h=0.005$')
plt.plot(epsilon, error[:, 4, 0], color='brown', marker='+', label='G12 $= \langle (-\Delta)^{-1}u_\epsilon(f_i),(-\Delta)^{-1}(f_j)$' + r'$\rangle$' + '$ h=0.001$')
plt.plot(epsilon, error[:, 4, 1], color='brown', marker='+', label='G12 $= \langle (-\Delta)^{-1}u_\epsilon(f_i),(-\Delta)^{-1}(f_j)$' + r'$\rangle$' + '$ h=0.005$')
plt.plot(epsilon, error[:, 5, 0], color='black', marker='+', label='G2 $= \langle (-\Delta)^{-1}u_\epsilon(f_i),(-\Delta)^{-1}u_\epsilon(f_j)$' + r'$\rangle$' + '$ h=0.001$')
plt.plot(epsilon, error[:, 5, 1], color='black', marker='+', label='G2 $= \langle (-\Delta)^{-1}u_\epsilon(f_i),(-\Delta)^{-1}u_\epsilon(f_j)$' + r'$\rangle$' + '$ h=0.005$')
plt.legend()
plt.title("Relativ error, $|||G^\star - G_\epsilon|||/|||G^\star|||$")
plt.xlabel("$\epsilon$")
plt.ylabel("Relativ error")
plt.xscale("log")
plt.yscale("log")
plt.show()