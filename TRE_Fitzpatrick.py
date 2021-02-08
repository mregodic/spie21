# -*- coding: utf-8 -*-
"""
Fitzpatrick target registration error

TRE**2 = (FLE**2 / N) * (1 + 1/3 * np.sum(r));

"""

import numpy as np

# isotropic FLE error
FLE = 0.8

K = 3

x0 = np.array([[-27.72, -155.74, -187.139],
              [-7.69921, -150.066, -185.377],
              [-21.1434, -144.553, -185.662],
              [ -21.6896, -149.304, -169.095]])

r0 = np.array([[0., 0., 0.]])

# Cochlea -  left ear
#r0 = np.array([[21.6132, -121.289, -164.487]]) 

# Cochlea - right ear
#r0 = np.array([[-57.6696, -112.693, -168.925]]) 

# foramina of luschka - left ear
#r0 = np.array([[-4.33348, -109.48, -173.807]]) 

# foramina of luschka - right ear
#r0 = np.array([[-28.5291, -107.98, -176.026]]) 

# Thalamus left ear
#r0 = np.array([[-8.16883, -130.682, -109.902]]) 

# Thalamus right ear
#r0 = np.array([[-41.8963, -130.682, -112.565]])

# Hypothalamus
#r0 = np.array([[-22.3115, -131.756, -134.754]]) 

N = np.size(x0, axis = 0)

print("fiducials:", N)

x_mean = np.mean(x0, axis = 0)
print("fiducial mean:\n", x_mean)

# demean markers
x = x0 - x_mean

# demean tip
r = r0 - x_mean

print ("\ndemean x:\n", x)
print ("\ndemean tip:\n", r)

# calculate rms distance of fiducials from each principal axis

# transform markers into principal axes.
U, Lambda, V = np.linalg.svd(x)

V = np.transpose(V)

print ("\nU:\n", U)
print ("\nLambda:\n", Lambda)
print ("V:\n", V)

x = x @ V

r = r @ V
 
print ("\nprinciple distance x :\n", x)
print ("\nprinciple distance tip:\n", r)

f = np.array([0., 0., 0.])
f[0] = np.sqrt(np.mean(x[:,1]**2 + x[:,2]**2))
f[1] = np.sqrt(np.mean(x[:,0]**2 + x[:,2]**2))
f[2] = np.sqrt(np.mean(x[:,0]**2 + x[:,1]**2))

print("\nfiducial distance:\n", f)

d = np.array([0., 0., 0.])
d[0] = np.sqrt(np.mean(r[:,1]**2 + r[:,2]**2))
d[1] = np.sqrt(np.mean(r[:,0]**2 + r[:,2]**2))
d[2] = np.sqrt(np.mean((r[:,0]**2) + r[:,1]**2))

print("\ntarget distance:\n", d)

# get ratios
r = (d/f)**2
print ("\nratio d/f:", r)

TRE2 = (FLE**2 / N) * (1 + 1/3 * np.sum(r));

TRE = np.sqrt(TRE2);

FRE1 = np.sqrt((N - 1/2 * (K + 1)) * FLE**2)
FRE2 = np.sqrt((1 - 2.0/N) * FLE**2)

print("Fitz TRE:", TRE)
print("Fitz FRE 1:", FRE1)
print("Fitz FRE 2:", FRE2)

