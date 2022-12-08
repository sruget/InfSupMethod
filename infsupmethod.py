# author : Simon Ruget
# This file contains functions related to the algorithm inf sup

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from loadMatrix import *

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


def main(Winit=0., itermax=10, threshold=10**(-10)):
    # Loading the Matrix G0, G1, G2
    G0 = assembleG0('Precomputation_G01.txt', 'Precomputation_G02.txt', 'Precomputation_G03.txt')
    G1 = assembleG1('Precomputation_G11.txt', 'Precomputation_G12.txt')
    G2 = assembleG2('Precomputation_G2.txt')
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

main()