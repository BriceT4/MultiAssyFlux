#!/usr/bin/python
# inp_MultiAssyFlux_Setup. py
# Input file for MultiAssyFlux.py project
# UFL ENU6106 Spring 2023 Term Project
# Brice Turner, 2023

D = 0.011058*100 # fuel rod diameter (cm) 
Pitch = 0.014385*100 # fuel rod pitch (cm)
L = 0.244545*100 # Assembly length (cm)

num_mesh = 4 # number of meshes per unit cell

mat_1 = 'Uranium' # 'Uranium' or 'MOX'
mat_2 = 'Uranium' # 'Uranium' or 'MOX'
bc_1 = 'Pure reflector' # 'Vacuum', 'Pure reflector', or 'Water reflector'
bc_2 = 'Pure reflector' # 'Vacuum', 'Pure reflector', or 'Water reflector'

model = 'Monte Carlo' # Can only be 'Monte Carlo' for now
x_sections_set = 'xs\\x_sections_set_A.csv' # last letter can be A or B

# number of generations
num_gen = 2
# number of histories (particles)
num_particles = 10
