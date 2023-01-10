# author : Simon Ruget
# This file contains functions related to the algorithm inf sup

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from loadMatrix import *
plt.rcParams['text.usetex'] = True

#################
### Constants ###
#################
ITERMAX = 80

##################
### Algorithme ###
##################

mu = 0. ## Be careful, mu !=0 implies to change the way the basic component of G0, G1, G2 are computed (u_eps).
alpha = 0.1

def sup(G0, G1, G2, W):
    """Compute the solution f = \sum c_p f_p solution of the sup problem.
    (c_p)_p is stored in eig list"""
    G = G0 + (W+mu)*G1 + (W+mu)*(W+mu)*G2
    eigval, eigvect = eigh(G) #find the eigenvalue, eigenvector of G using LU decomposition
    #print("eigval : ", eigval)
    #print("eigvect : ", eigvect)
    argmax = np.argmax(np.abs(eigval))
    eig = eigvect[:,argmax]
    norm = np.sqrt(np.dot(eig, eig))
    eig = eig/norm
    #print(argmax)
    #print(np.dot(eigvect[:,argmax], eigvect[:,argmax]))
    return(eig)

def inf(G0, G1, G2, f):
    """Compute the solution W of the inf problem."""
    num = np.dot(f, np.dot(G1, f))
    denum = np.dot(f, np.dot(G2, f))
    return(-num/(2*denum) - mu)


def main(filenameG01, filenameG02, filenameG03, filenameG11, filenameG12, filenameG2, Winit=0., itermax=ITERMAX, threshold=10**(-20), P=10):
    """Compute an approximation of the homogenized potential through alternate deviation method (alternance of inf-sup problem resolution)"""
    # Loading the Matrix G0, G1, G2
    G0 = assembleG0(filenameG01, filenameG02, filenameG03)[:P, :P]
    G1 = assembleG1(filenameG11, filenameG12)[:P, :P]
    G2 = assembleG2(filenameG2)[:P, :P]
    #P = np.shape(G0)[0] # pass as an argument

    #Initialization
    W = Winit
    W_int = 0.
    Val_av = 15.
    Val_ap = -15.
    f=np.ones(P)/np.sqrt(P)
    #f[0] = 1.
    p = 0
    ftest = np.zeros(P)
    ftest[0] = 1

    # Monitoring Convergence quality
    residu = np.zeros(itermax) #store np.abs(W_k - W_end)

    while (p<itermax and np.abs(Val_av-Val_ap)>threshold):
        print()
        print("p : ", p)
        print("W : ", W)
        print("fsup : ", f)
        print("val(f,w) : ", np.dot(f, np.dot(G0 + (W+mu)*G1 + (W+mu)*(W+mu)*G2, f)))

        f = sup(G0, G1, G2, W)

        # Checking the validity of supremum
        if (np.dot(f, np.dot(G0 + (W+mu)*G1 + (W+mu)*(W+mu)*G2, f)) < np.dot(ftest, np.dot(G0 + (W+mu)*G1 + (W+mu)*(W+mu)*G2, ftest))) :
            print("the sup is wrong, p=", p)
        
        W_int = inf(G0, G1, G2, f)
        print("Wint : ", W_int)
        W =(1-alpha)*W + alpha*W_int
        Val_ap, Val_av = np.dot(f, np.dot(G0 + (W+mu)*G1 + (W+mu)*(W+mu)*G2, f)), Val_ap
        print("Valap : ", Val_ap)
        residu[p] = W
        p+=1
    Wstar = -1./(8*np.pi*np.pi)
    residu = np.abs(residu-residu[len(residu)-1])/np.abs(residu[len(residu)-1])
    #residu = (residu-Wstar)/np.abs(Wstar)
    #print("P = ", P, " W_bar =", W)
    return(Val_ap, f, W, residu)








filenameG01 = "./results/"+"12_12_2022"+"/Precomputation_G01_eps"+"032"+"_"+"121222"+".txt"
filenameG02 = "./results/"+"12_12_2022"+"/Precomputation_G02_eps"+"032"+"_"+"121222"+".txt"
filenameG03 = "./results/"+"12_12_2022"+"/Precomputation_G03_eps"+"032"+"_"+"121222"+".txt"
filenameG11 = "./results/"+"12_12_2022"+"/Precomputation_G11_eps"+"032"+"_"+"121222"+".txt"
filenameG12 = "./results/"+"12_12_2022"+"/Precomputation_G12_eps"+"032"+"_"+"121222"+".txt"
filenameG2  = "./results/"+"12_12_2022"+"/Precomputation_G2_eps"+"032"+"_"+"121222"+".txt"
V_bar = main(filenameG01, filenameG02, filenameG03, filenameG11, filenameG12, filenameG2, P=10)[2]
print()
print()
print()
print()
print()
print()

V_bar = main(filenameG01, filenameG02, filenameG03, filenameG11, filenameG12, filenameG2, P=1)[2]







############
### Plot ###
############

## Plot Convergence of infsup method
if (__name__ == "__main__" and 1==1) :
    V_star = -1./(8*np.pi*np.pi)
    datedir = ["24_12_2022"] #["12_12_2022"]#, "11_12_2022", "10_12_2022"]
    datefile = ["241222"]#["121222"]
    h_value = [0.001] #h value is linked to date of file
    N_date = len(datedir)
    epslist = ["050", "048", "046", "044", "042", "040", "038", "036", "034", "033", "032", "030", "028", "026", "025", "024", "022", "020", "018", "016", "014", "0125", "012", "011", "010"] #["064", "032", "016", "008", "004"]
    epsilon = np.array([0.50, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36, 0.34, 1./3, 0.32, 0.30, 0.28, 0.26, 0.25, 0.24, 0.22, 0.20, 0.18, 1./6, 0.14, 0.125, 0.12, 1./9, 0.10]) #np.array([0.64, 0.32, 0.16, 0.08, 0.04])
    list_P = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    N_P = len(list_P)
    N_eps = len(epsilon)
    error = np.zeros((N_eps, N_date, N_P))
    for p in range(N_P) :
        for d in range(N_date) :
            for e, eps in enumerate(epslist):
                print(eps, p)
                filenameG01 = "./results/"+str(datedir[d])+"/Precomputation_G01_eps"+str(eps)+"_"+str(datefile[d])+".txt"
                filenameG02 = "./results/"+str(datedir[d])+"/Precomputation_G02_eps"+str(eps)+"_"+str(datefile[d])+".txt"
                filenameG03 = "./results/"+str(datedir[d])+"/Precomputation_G03_eps"+str(eps)+"_"+str(datefile[d])+".txt"
                filenameG11 = "./results/"+str(datedir[d])+"/Precomputation_G11_eps"+str(eps)+"_"+str(datefile[d])+".txt"
                filenameG12 = "./results/"+str(datedir[d])+"/Precomputation_G12_eps"+str(eps)+"_"+str(datefile[d])+".txt"
                filenameG2  = "./results/"+str(datedir[d])+"/Precomputation_G2_eps"+str(eps)+"_"+str(datefile[d])+".txt"
                V_bar = main(filenameG01, filenameG02, filenameG03, filenameG11, filenameG12, filenameG2, P=list_P[p])[2]
                error[e, d, p] = np.abs(V_bar-V_star)/np.abs(V_star)


    #for p in range(N_P) :
    #    for d in range(N_date) :
    plt.plot(epsilon, error[:, 0, 0], color='red', marker='+', label='$P=$'+str(list_P[0]))
    plt.plot(epsilon, error[:, 0, 5], color='green', marker='+', label='$P=$'+str(list_P[5]))
    plt.plot(epsilon, error[:, 0, 9], color='blue', marker='+', label='$P=$'+str(list_P[9]))
    plt.plot(epsilon, 40.*epsilon**2, color='black', marker='+', label='$y \propto x^2$')
    #plt.plot(epsilon, error[:, 1, 0], color='blue', marker='+', label='$h=0.005$')
    #plt.plot(epsilon, error[:, 2, 0], color='green', marker='+', label='$h=0.001$')
    plt.legend()
    plt.title("Relativ error, $|V^\star - \overline{V}_\epsilon|/|V^\star|$")
    plt.xlabel("$\epsilon$")
    plt.ylabel("Relativ error")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()


## Write BarVeps
if (__name__ == "__main__" and 1==0) :
    datedir = ["24_12_2022"] #["12_12_2022"]#, "11_12_2022", "10_12_2022"]
    datefile = ["241222"]#["121222"]
    h_value = [0.001] #h value is linked to date of file
    N_date = len(datedir)
    epslist = ["050", "048", "046", "044", "042", "040", "038", "036", "034", "033", "032", "030", "028", "026", "025", "024", "022", "020", "018", "016", "014", "0125", "012", "011", "010"] #["064", "032", "016", "008", "004"]
    epsilon = np.array([0.50, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36, 0.34, 1./3, 0.32, 0.30, 0.28, 0.26, 0.25, 0.24, 0.22, 0.20, 0.18, 1./6, 0.14, 0.125, 0.12, 1./9, 0.10]) #np.array([0.64, 0.32, 0.16, 0.08, 0.04])
    N_eps = len(epsilon)
    list_P = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    N_P = len(list_P)

    filenamebarV = "BarVeps_"+str(datefile[0])+".txt"
    with open(filenamebarV, 'w') as f:
        f.write('Approximation bar{V} of Potential V_\star computed with InfSupMethod (there is various choice of p possible to compute the sup solution)\n')
        f.write('eps, p=1, p=2, p=3, p=4, p=5, p=6, p=7, p=8, p=9, p=10\n')
        for d in range(N_date) :
            for e, eps in enumerate(epslist):
                f.write(str(epsilon[e]))
                for p in range(N_P) :
                    print(eps, p)
                    filenameG01 = "./results/"+str(datedir[d])+"/Precomputation_G01_eps"+str(eps)+"_"+str(datefile[d])+".txt"
                    filenameG02 = "./results/"+str(datedir[d])+"/Precomputation_G02_eps"+str(eps)+"_"+str(datefile[d])+".txt"
                    filenameG03 = "./results/"+str(datedir[d])+"/Precomputation_G03_eps"+str(eps)+"_"+str(datefile[d])+".txt"
                    filenameG11 = "./results/"+str(datedir[d])+"/Precomputation_G11_eps"+str(eps)+"_"+str(datefile[d])+".txt"
                    filenameG12 = "./results/"+str(datedir[d])+"/Precomputation_G12_eps"+str(eps)+"_"+str(datefile[d])+".txt"
                    filenameG2  = "./results/"+str(datedir[d])+"/Precomputation_G2_eps"+str(eps)+"_"+str(datefile[d])+".txt"
                    V_bar = main(filenameG01, filenameG02, filenameG03, filenameG11, filenameG12, filenameG2, P=list_P[p])[2]
                    f.write(',' + str(V_bar))
                f.write('\n')











## Monitoring Convergence quality
if (__name__ == "__main__" and 1==0) :
    V_star = -1./(8*np.pi*np.pi)
    datedir = ["12_12_2022"]#"11_12_2022", "10_12_2022"]
    datefile = ["121222"]#"111222", "101222"]
    N_date = len(datedir)
    epslist = ["064", "032", "016", "008"]
    epsilon = np.array([0.64, 0.32, 0.16, 0.08])
    N_eps = len(epsilon)
    list_P = [1, 6, 10]
    N_P = len(list_P)
    error = np.zeros((N_eps, N_date, N_P, ITERMAX-1))
    for d in range(N_date) :
        for e, eps in enumerate(epslist):
            print("eps =", eps)
            for p in range(N_P) :
                print("p =", list_P[p])
                filenameG01 = "./results/"+str(datedir[d])+"/Precomputation_G01_eps"+str(eps)+"_"+str(datefile[d])+".txt"
                filenameG02 = "./results/"+str(datedir[d])+"/Precomputation_G02_eps"+str(eps)+"_"+str(datefile[d])+".txt"
                filenameG03 = "./results/"+str(datedir[d])+"/Precomputation_G03_eps"+str(eps)+"_"+str(datefile[d])+".txt"
                filenameG11 = "./results/"+str(datedir[d])+"/Precomputation_G11_eps"+str(eps)+"_"+str(datefile[d])+".txt"
                filenameG12 = "./results/"+str(datedir[d])+"/Precomputation_G12_eps"+str(eps)+"_"+str(datefile[d])+".txt"
                filenameG2  = "./results/"+str(datedir[d])+"/Precomputation_G2_eps"+str(eps)+"_"+str(datefile[d])+".txt"
                error[e, d, p, :] = main(filenameG01, filenameG02, filenameG03, filenameG11, filenameG12, filenameG2, P=list_P[p])[3][1:ITERMAX]
                print("W : ", main(filenameG01, filenameG02, filenameG03, filenameG11, filenameG12, filenameG2, P=list_P[p])[2])
                fsup = main(filenameG01, filenameG02, filenameG03, filenameG11, filenameG12, filenameG2, P=list_P[p])[1]
                print("fsup : ", fsup)
                print("norm fsup : ", np.dot(fsup, fsup))
                print()


    plt.plot(np.arange(1,ITERMAX), error[1, 0, 0,:], color='red', marker='+', label='$\epsilon =$' + str(epsilon[1]) + ', P =' + str(list_P[0]))
    plt.plot(np.arange(1,ITERMAX), error[2, 0, 1,:], color='green', marker='+', label='$\epsilon =$' + str(epsilon[2]) + ', P =' + str(list_P[1]))
    plt.plot(np.arange(1,ITERMAX), error[3, 0, 2,:], color='blue', marker='+', label='$\epsilon =$' + str(epsilon[3]) + ', P =' + str(list_P[2]))
    plt.plot(np.arange(1,ITERMAX), error[1, 0, 0,:], color='orange', marker='v', label='$\epsilon =$' + str(epsilon[1]) + ', P =' + str(list_P[0]))
    plt.plot(np.arange(1,ITERMAX), error[2, 0, 1,:], color='brown', marker='v', label='$\epsilon =$' + str(epsilon[2]) + 'P =' + str(list_P[1]))
    plt.plot(np.arange(1,ITERMAX), error[3, 0, 2,:], color='pink', marker='v', label='$\epsilon =$' + str(epsilon[3]) + 'P =' + str(list_P[2]))
    #plt.plot(np.arange(1,ITERMAX), error[2, 0, :], color='blue', marker='+', label='$\epsilon =$' + str(epsilon[2]))
    #plt.plot(np.arange(1,ITERMAX), error[3, 0, :], color='green', marker='+', label='$\epsilon =$' + str(epsilon[3]))
    #plt.plot(np.arange(1,ITERMAX), 0.01*np.ones(ITERMAX-1), color='black', label='1\%')
    plt.legend()
    plt.title("Monitoring Convergence for various $\epsilon$")
    plt.xlabel("iterations k")
    plt.ylabel("$(\overline{V}_\epsilon^k - V_\star)/|V_\star|$")
    #plt.xscale("log")
    #plt.yscale("log")
    plt.show()










## Plot |Functionnal_P - Functionnal_P-1|
if (__name__ == "__main__" and 1==0) :
    datedir = ["12_12_2022"]
    datefile = ["111222"]
    N_date = len(datedir)
    epslist = ["032", "016", "008"]
    epsilon = np.array([0.32, 0.16, 0.08])
    N_eps = len(epsilon)
    list_P = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    N_P = len(list_P)
    functionnal = np.zeros((N_eps, N_P-1))
    for d in range(N_date) :
        for e, eps in enumerate(epslist):
            filenameG01 = "./results/"+str(datedir[d])+"/Precomputation_G01_eps"+str(eps)+"_"+str(datefile[d])+".txt"
            filenameG02 = "./results/"+str(datedir[d])+"/Precomputation_G02_eps"+str(eps)+"_"+str(datefile[d])+".txt"
            filenameG03 = "./results/"+str(datedir[d])+"/Precomputation_G03_eps"+str(eps)+"_"+str(datefile[d])+".txt"
            filenameG11 = "./results/"+str(datedir[d])+"/Precomputation_G11_eps"+str(eps)+"_"+str(datefile[d])+".txt"
            filenameG12 = "./results/"+str(datedir[d])+"/Precomputation_G12_eps"+str(eps)+"_"+str(datefile[d])+".txt"
            filenameG2  = "./results/"+str(datedir[d])+"/Precomputation_G2_eps"+str(eps)+"_"+str(datefile[d])+".txt"
            for p in range(1, N_P) :
                Phi_p = main(filenameG01, filenameG02, filenameG03, filenameG11, filenameG12, filenameG2, P=list_P[p])[0]
                Phi_q = main(filenameG01, filenameG02, filenameG03, filenameG11, filenameG12, filenameG2, P=list_P[p-1])[0]
                functionnal[e, p-1] = np.abs(Phi_p - Phi_q)/np.abs(Phi_p)
    print(functionnal[0, :])
    print()
    print(functionnal[1, :])
    print()
    print(functionnal[2, :])
    print()
    plt.plot(list_P[1:N_P], functionnal[0, :], color='red', marker='+', label='$\epsilon =$' + str(epsilon[0]))
    plt.plot(list_P[1:N_P], functionnal[1, :], color='blue', marker='+', label='$\epsilon =$' + str(epsilon[1]))
    plt.plot(list_P[1:N_P], functionnal[2, :], color='green', marker='+', label='$\epsilon =$' + str(epsilon[2]))
    plt.legend()
    plt.title("Variation of Results in term of Number of eigenvalue")
    plt.xlabel("Number of eigenvalue P")
    plt.ylabel("$|\Phi_\epsilon^P - \Phi_\epsilon^{P-1}|/|\Phi_\epsilon^{P}|$")
    #plt.yscale("log")
    plt.show()

def convexityofsup(filenameG01, filenameG02, filenameG03, filenameG11, filenameG12, filenameG2, Winit=0., P=10):
    """Plot the function W rightarrow sup_f \Phi_\epsilon(W, f) to show its convexity"""
    # Loading the Matrix G0, G1, G2
    G0 = assembleG0(filenameG01, filenameG02, filenameG03)[:P, :P]
    G1 = assembleG1(filenameG11, filenameG12)[:P, :P]
    G2 = assembleG2(filenameG2)[:P, :P]

    W_list = np.linspace(-1, 1, 50)
    convexe_func = np.zeros(50)
    for i, W in enumerate(W_list) :
        f = sup(G0, G1, G2, W)
        convexe_func[i] = np.dot(f, np.dot(G0 + (W+mu)*G1 + (W+mu)*(W+mu)*G2, f))
    plt.plot(W_list, convexe_func)
    plt.legend()
    plt.title('$\Phi_\epsilon : V$' + r'$\rightarrow$' + '$\sup_f \Big(\phi_\epsilon(V, f)\Big)$, $\epsilon = 0.64$')
    plt.xlabel("$V$")
    plt.ylabel("$\Phi_\epsilon(V)$")
    plt.show()

## Plot convexity
if (__name__ == "__main__" and 1==0) :
    datedir = "11_12_2022"
    datefile = "111222"
    eps = "064" #change in title of plot in convexityofsup
    filenameG01 = "./results/"+str(datedir)+"/Precomputation_G01_eps"+str(eps)+"_"+str(datefile)+".txt"
    filenameG02 = "./results/"+str(datedir)+"/Precomputation_G02_eps"+str(eps)+"_"+str(datefile)+".txt"
    filenameG03 = "./results/"+str(datedir)+"/Precomputation_G03_eps"+str(eps)+"_"+str(datefile)+".txt"
    filenameG11 = "./results/"+str(datedir)+"/Precomputation_G11_eps"+str(eps)+"_"+str(datefile)+".txt"
    filenameG12 = "./results/"+str(datedir)+"/Precomputation_G12_eps"+str(eps)+"_"+str(datefile)+".txt"
    filenameG2 = "./results/"+str(datedir)+"/Precomputation_G2_eps"+str(eps)+"_"+str(datefile)+".txt"
    convexityofsup(filenameG01, filenameG02, filenameG03, filenameG11, filenameG12, filenameG2)