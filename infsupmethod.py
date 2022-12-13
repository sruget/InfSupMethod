# author : Simon Ruget
# This file contains functions related to the algorithm inf sup

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from loadMatrix import *

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


def main(filenameG01, filenameG02, filenameG03, filenameG11, filenameG12, filenameG2, Winit=0., itermax=100, threshold=10**(-20)):
    # Loading the Matrix G0, G1, G2
    G0 = assembleG0(filenameG01, filenameG02, filenameG03)
    G1 = assembleG1(filenameG11, filenameG12)
    G2 = assembleG2(filenameG2)
    P = np.shape(G0)[0]

    #Initialization
    W = Winit
    W_int = 0.
    Val_av = 15.
    Val_ap = -15.
    f=np.zeros(P)
    f[0] = 1.
    p = 0

    while (p<itermax and np.abs(Val_av-Val_ap)>threshold):
        print(p)
        print("f : ", f)
        print("Val of Pb : ", Val_ap)
        print("W : ", W)
        print()
        f = sup(G0, G1, G2, W)
        W_int = inf(G0, G1, G2, f)
        W =(1-alpha)*W + alpha*W_int
        Val_ap, Val_av = np.dot(f, np.dot(G0 + (W+mu)*G1 + (W+mu)*(W+mu)*G2, f)), Val_ap
        p+=1
    return(Val_ap, f, W)

########################
### Plot Convergence ###
########################

plt.rcParams['text.usetex'] = True

V_star = -1./(8*np.pi*np.pi)
datedir = ["11_12_2022", "10_12_2022"]
datefile = ["111222", "101222"]
N_date = len(datedir)
epslist = ["064", "032", "016", "008"]
epsilon = np.array([0.64, 0.32, 0.16, 0.08])
N_eps = len(epsilon)
error = np.zeros((N_eps, N_date))
for d in range(N_date) :
    for e, eps in enumerate(epslist):
        filenameG01 = "./results/"+str(datedir[d])+"/Precomputation_G01_eps"+str(eps)+"_"+str(datefile[d])+".txt"
        filenameG02 = "./results/"+str(datedir[d])+"/Precomputation_G02_eps"+str(eps)+"_"+str(datefile[d])+".txt"
        filenameG03 = "./results/"+str(datedir[d])+"/Precomputation_G03_eps"+str(eps)+"_"+str(datefile[d])+".txt"
        filenameG11 = "./results/"+str(datedir[d])+"/Precomputation_G11_eps"+str(eps)+"_"+str(datefile[d])+".txt"
        filenameG12 = "./results/"+str(datedir[d])+"/Precomputation_G12_eps"+str(eps)+"_"+str(datefile[d])+".txt"
        filenameG2  = "./results/"+str(datedir[d])+"/Precomputation_G2_eps"+str(eps)+"_"+str(datefile[d])+".txt"
        V_bar = main(filenameG01, filenameG02, filenameG03, filenameG11, filenameG12, filenameG2)[2]
        error[e, d] = np.abs(V_bar-V_star)/np.abs(V_star)

plt.plot(epsilon, error[:, 0], color='red', marker='+', label='$h=0.001$')
plt.plot(epsilon, error[:, 1], color='blue', marker='+', label='$h=0.005$')
plt.legend()
plt.title("Relativ error, $|V^\star - \overline{V}|/|V^\star|$")
plt.xlabel("$\epsilon$")
plt.ylabel("Relativ error")
plt.xscale("log")
plt.yscale("log")
plt.show()