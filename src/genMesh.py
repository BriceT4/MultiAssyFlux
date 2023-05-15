#!/usr/bin/python
# genMesh.py
# A MultiAssyFlux module
# UFL ENU6106 Spring 2023 Term Project
# Brice Turner



# BEGIN: IMPORT MODULES ######################################################
import numpy as np
import os
import pandas as pd
import time

time_start_genMesh = time.time()
# END:   IMPORT MODULES ######################################################



# BEGIN: DEFINE FUNCTION #####################################################
def genMesh(input_file):

    # BEGIN: GENERATE MESH OF IDs ############################################

    # Material IDs:
    # water = 0
    # U_fuel = 1
    # MOX = 2
    # control_rod = 3
    # air = 4

    num_mesh = input_file.num_mesh # number of meshes per cell of fuel or moderator
    thick_cell_fuel = input_file.D # fuel cell thickness (m)
    thick_cell_mod = input_file.Pitch - thick_cell_fuel # moderator cell thickness (m)
    thick_cell_mod_ends = thick_cell_mod/2 # moderator "half-cell" thickness (m)
    Delta_x_fuel = thick_cell_fuel/num_mesh
    Delta_x_mod = thick_cell_mod/num_mesh
    Delta_x_mod_ends = thick_cell_mod_ends/num_mesh
    num_gen = input_file.num_gen
    num_particles = input_file.num_particles

    # create arrays of ID's for each fuel type
    # mat_U =     np.array([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
    #             0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
    # mat_MOX =   np.array([0,2,0,2,0,2,0,2,0,2,0,2,0,2,0,2,0,2,
    #             0,2,0,2,0,2,0,2,0,2,0,2,0,2,0,2,0])
    mat_U = np.array([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,
                0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0])
    mat_MOX = np.array([0,2,0,2,0,2,0,2,0,2,0,2,0,2,0,2,0,2,
                0,2,0,2,0,2,0,2,0,2,0,2,0,2,0,2,0])
    mat_water_ref = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    mat_vac_ref = np.array([4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
                              4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4])


    # apply array to new mesh based on input_fileut file
    if input_file.mat_1 == 'Uranium':
        mesh_mat_1 = mat_U
    elif input_file.mat_1 == 'MOX':
        mesh_mat_1 = mat_MOX

    if input_file.mat_2 == 'Uranium':
        mesh_mat_2 = mat_U
    elif input_file.mat_2 == 'MOX':
        mesh_mat_2 = mat_MOX

    # create dataframe of meshes side-by-side
    if input_file.bc_1 == 'Water reflector' and input_file.bc_2 == 'Water reflector':
        mesh_mat_tot = np.append(np.append(mat_water_ref,
                                           np.append(mesh_mat_1,
                                                     mesh_mat_2)), mat_water_ref)
    elif input_file.bc_1 == 'Water reflector':
        mesh_mat_tot = np.append(mat_water_ref,
                                 np.append(mesh_mat_1,
                                           mesh_mat_2))
    elif input_file.bc_2 == 'Water reflector':
        mesh_mat_tot = np.append(np.append(mesh_mat_1,
                                           mesh_mat_2), mat_water_ref)
    else:
        mesh_mat_tot = np.append(mesh_mat_1, mesh_mat_2)

    mesh_mat = np.repeat(mesh_mat_tot, num_mesh)
    mesh = pd.DataFrame({'mat_ID':mesh_mat})

    # add string names corresponding to IDs to mesh
    def append_names(val):
        if val == 0:
            return 'water'
        elif val == 1:
            return 'U_fuel'
        elif val == 2:
            return 'MOX'
        elif val == 3:
            return 'control_rod'
        else:
            return 'air'

    mesh['material'] = mesh['mat_ID'].apply(append_names)
    mesh = mesh.transpose()
    # END:   GENERATE MESH OF IDs ############################################



    # BEGIN: ADD CROSS SECTIONS TO MESH DATA #################################
    #import cross section data
    x_secs_all = pd.read_csv(input_file.x_sections_set, index_col = 0)
    #initialize
    mesh_x_secs = pd.DataFrame()

    # apply cross sections to each mesh
    for m in mesh.loc['mat_ID',:]:
        if m == 0:
            x_secs = x_secs_all.loc['water', :].T
        elif m == 1:
            x_secs = x_secs_all.loc['U_fuel', :].T
        elif m == 2:
            x_secs = x_secs_all.loc['MOX', :].T
        if input_file.x_sections_set == 'x_sections_B.csv':
            if m == 3:
                x_secs = x_secs_all.loc['control_rod', :].T
            elif m == 4:
                x_secs = x_secs_all.loc['air', :].T

        mesh_x_secs = pd.concat([mesh_x_secs, x_secs], axis = 1,
                                ignore_index = True)
    mesh = pd.concat([mesh, mesh_x_secs])
    # END:   ADD CROSS SECTIONS TO MESH DATA #################################



    # BEGIN: ADD X POSITIONS TO MESH DATA ####################################
    mesh_loc = pd.DataFrame()
    loc_current = 0
    
    for m in np.arange(0, np.size(mesh, 1)):
        # define thicknesses and DeltaX's:
        if m < num_mesh:
            # you're in first cells
            Delta_x = Delta_x_mod_ends
            loc_left = loc_current
            loc_center = loc_left + 0.5*Delta_x_mod_ends
            loc_right = loc_center + 0.5*Delta_x_mod_ends

        elif m < (np.size(mesh,1)/2 - num_mesh):
            # you're in a full-size cell in first assembly                    
            if mesh.loc['mat_ID', m] ==  0: # water
                Delta_x = Delta_x_mod
                loc_left = loc_current
                loc_center = loc_left + 0.5*Delta_x_mod
                loc_right = loc_center + 0.5*Delta_x_mod

            else: # fuel or control rod
                Delta_x = Delta_x_fuel
                loc_left = loc_current
                loc_center = loc_left + 0.5*Delta_x_fuel
                loc_right = loc_center + 0.5*Delta_x_fuel

        elif m < (np.size(mesh,1)/2 + num_mesh):
            # your where the assemblies meet
            Delta_x = Delta_x_mod_ends
            loc_left = loc_current
            loc_center = loc_left + 0.5*Delta_x_mod_ends
            loc_right = loc_center + 0.5*Delta_x_mod_ends

        elif m < (np.size(mesh,1) - num_mesh):
            # you're in a full-size cell in second assembly   
            if mesh.loc['mat_ID', m] ==  0: # water
                Delta_x = Delta_x_mod
                loc_left = loc_current
                loc_center = loc_left + 0.5*Delta_x_mod
                loc_right = loc_center + 0.5*Delta_x_mod

            else: # fuel or control rod
                Delta_x = Delta_x_fuel
                loc_left = loc_current
                loc_center = loc_left + 0.5*Delta_x_fuel
                loc_right = loc_center + 0.5*Delta_x_fuel
            
        else:
            # you're in last cell (half water)
            Delta_x = Delta_x_mod_ends
            loc_left = loc_current
            loc_center = loc_left + 0.5*Delta_x_mod_ends
            loc_right = loc_center + 0.5*Delta_x_mod_ends

        loc_current = loc_right
        locs = pd.DataFrame({'locs_left':[loc_left],
                            'locs_center':[loc_center],
                            'locs_right':[loc_right],
                            'Delta_x':[Delta_x]}).T
        mesh_loc = pd.concat([mesh_loc, locs] , axis = 1, ignore_index = True)

    mesh = pd.concat([mesh, mesh_loc])
    # END:   ADD X POSITIONS TO MESH DATA ####################################



    # BEGIN: CREATE FUEL-ONLY MESH ###########################################
    # break down mesh into new ones by material (Useful for neutron birth)
    mesh_fuel = mesh
    for c in mesh_fuel.columns:
        if mesh_fuel.loc['mat_ID', c] == 0 or mesh_fuel.loc['mat_ID', c] == 3:
            mesh_fuel = mesh_fuel.drop(c, axis = 1)
    cols_old = pd.DataFrame(mesh_fuel.columns).T
    cols_old.rename_axis('cols_old')
    mesh_fuel.columns = range(mesh_fuel.shape[1])
    mesh_fuel.loc['cols_old',:] = cols_old.loc[0,:]
    # END:   CREATE FUEL-ONLY MESH ###########################################



    # BEGIN: OUTPUTS #########################################################
    # print(mesh)
    # print(mesh_fuel)

    timestr = time.strftime('%Y%m%d_%H%M%S')
    filename = f'MultiAssyFlux_mesh_{timestr}.csv'
    dir_output = f'outputs\\MultiAssyFlux_outputs_MC_g{num_gen}_n{num_particles}_{timestr}'
    os.makedirs(dir_output)
    fp = os.path.join(dir_output, filename)
    mesh.to_csv(fp)

    time_elapsed_genMesh = time.time() - time_start_genMesh

    print(f"""
##########################################################################
Mesh generation complete.
Mesh exported to {filename}
in \{dir_output}\ 
Run time: {time_elapsed_genMesh:.3f} s.
##########################################################################
    """)
    # END:   OUTPUTS #########################################################
    
    return mesh, mesh_fuel, dir_output
# END:   DEFINE FUNCTION #####################################################

