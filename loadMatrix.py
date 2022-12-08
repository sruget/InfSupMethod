# author : Simon Ruget
# This file contains function to load matrixs G0, G1, G2

import numpy as np
import matplotlib.pyplot as plt

def loadMatrix(filename):
    """This function loads the matrix store in file of name filename.
    The structure of the file is the following : 
    Parameters
    eps = 0.1
    h = 0.01
    P = 20
    M =
    ..., ..., ..., ...\n
    ..., ..., ..., ...\n
    ..., ..., ..., ...\n
    """

    with open(filename, 'r') as f:
        next(f)
        next(f)
        next(f)
        P = int(f.readline()[4:6])
        next(f)

        M = np.zeros((P, P))
        for i, line in enumerate(f):
            l = line[:-1]
            value = l.split(",")
            for j in range(P):
                M[i,j] =float(value[j])
    f.close()
    return(M)

def assembleG0(filenameG01, filenameG02, filenameG03):
    """This function assemble the matrix G0 from the matrix ( u_eps(f_i) * u_eps(f_j) )_ij 
    stored in filenameG01, ( u_eps(f_i) * (-Lap^-1)(f_j) )_ij stored in filenameG02 and 
    the matrix ( (-Lap^-1)(f_i) * (-Lap^-1)(f_j) )_ij stored in filenameG03"""
    G01 = loadMatrix(filenameG01)
    G02 = loadMatrix(filenameG02)
    G03 = loadMatrix(filenameG03)
    G0 = G01 - G02 - np.transpose(G02) + G03
    print(G0)
    return(G0)

def assembleG1(filenameG11, filenameG12):
    """This function assemble the matrix G1 from the matrix ( (-Lap^-1)(u_eps(f_i)) * u_eps(f_j) )_ij 
    stored in filenameG11 and the matrix ( (-Lap^-1)(u_eps(f_i)) * (-Lap^-1)(f_j) )_ij stored in filenameG12"""
    G11 = loadMatrix(filenameG11)
    G12 = loadMatrix(filenameG12)
    G1 = G11 + np.transpose(G11) - G12 - np.transpose(G12)
    return(G1)

def assembleG2(filenameG2):
    """This function assemble the matrix G2 stored in file of name filenameG2."""
    return(loadMatrix(filenameG2))