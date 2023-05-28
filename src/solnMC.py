#!/usr/bin/python
# solnMC.py
# A MultiAssyFlux module
# (C) Brice Turner, 2023

import numpy as np
np.random.seed(0)
import os
import pandas as pd
import time


# BEGIN: MONTE CARLO SOLUTION FUNCTION #######################################
time_start_solnMC = time.time()
def solnMC(input_file, mesh, mesh_fuel, dir_output):

    size_mesh_fuel = len(mesh_fuel) - 1 # num cols - 1

    # BEGIN SAMPLE PARTICLE BIRTH POSITION ###################################
    def fn_det_pos_birth(mesh, mesh_fuel):
        loc_rand_mesh_fuel = np.random.randint(0, size_mesh_fuel)
        m = mesh_fuel[loc_rand_mesh_fuel].get('cols_old')

        mesh_vals = mesh[m]['locs_left'], mesh[m]['Delta_x']

        r = np.random.uniform()
        x = mesh_vals[0] + r * mesh_vals[1]

        return m, x
    # END:  SAMPLE PARTICLE BIRTH POSITION ###################################



    # BEGIN: DETERMINE INTERACTION TYPE ######################################
    def fn_det_mesh_interaction(mesh, m, n_E, mu, S, travel_dist):
        R = np.random.rand()

        # Sigma_tr = mesh.loc[f'Sigma_tr_{n_E}', m]
        Sigma_t = mesh[m][f'Sigma_t_{n_E}']
        Sigma_a = mesh[m][f'Sigma_a_{n_E}']
        Sigma_f = mesh[m][f'Sigma_f_{n_E}']
        Sigma_inscatter = mesh[m][f'Sigma_s-0_{n_E}-{n_E}']

        lim_upper_capture = (Sigma_a - Sigma_f)/Sigma_t 
        lim_upper_fission = (Sigma_a)/Sigma_t
        lim_upper_in_scatter = (Sigma_inscatter + Sigma_a)/Sigma_t

        if R < lim_upper_capture:
            # print('capture')
            n_exist = False
        elif lim_upper_capture <= R < lim_upper_fission:
            # print('fission')
            n_exist = False
        elif lim_upper_fission <= R < lim_upper_in_scatter:
            # print('in-scatter')
            n_exist = True
            mu = 2*np.random.rand() - 1
            S = -np.log(np.random.random())/Sigma_t
            travel_dist = mu*S
        elif lim_upper_in_scatter <= R:
            # print('down-scatter')
            n_exist = True
            n_E = 2
            mu = 2*np.random.rand() - 1
            S = -np.log(np.random.random())/mesh[m][f'Sigma_t_2']
            travel_dist = mu*S
        else:
            n_exist = False

        return n_exist, n_E, mu, S, travel_dist
    # END:   DETERMINE INTERACTION TYPE ######################################
    
    
    
    # BEGIN: DETERMINE BOUNDARY INTERACTION ##################################
    def fn_det_BC_interaction(BC, m, m_old, mu, travel_dist):
        if BC == 'Pure reflector' or BC == 'Water reflector':
            n_exist = True
            mu = -1*mu
            travel_dist = -1*travel_dist
            m = m_old # m when entering this is -1 b/c crossed BC
        else:
            n_exist = False
        # m = 0 # placeholder

        return n_exist, m, mu, travel_dist
    # END:   DETERMINE BOUNDARY INTERACTION ##################################



    # BEGIN: INITIALIZE VARIABLES ############################################
    data_gen_TL = np.zeros((2,len(mesh)))
    data_gen_J = np.zeros((2,len(mesh)))
    data_gen_Phi = np.zeros((2,len(mesh)))
    data_gen_Fi_t = np.zeros((2,len(mesh)))
    data_gen_ms_birth = np.zeros(len(mesh))
    data_tot_TL = np.zeros((2,len(mesh)))
    data_tot_J = np.zeros((2,len(mesh)))
    data_tot_Phi = np.zeros((2,len(mesh)))
    data_tot_Fi_t = np.transpose(np.zeros((len(mesh))))
    data_tot_ms_birth = np.zeros(len(mesh))
    num_gen = input_file.num_gen
    num_particles = input_file.num_particles
    ks = []
    k = 1
    # END:   INITIALIZE VARIABLES ############################################



    # BEGIN: FOR EACH GENERATION #############################################
    for g in range(num_gen):
        print(f'\t######## Generation: {g+1} ########')

        data_gen_TL = np.zeros((2,len(mesh)))
        data_gen_J = np.zeros((2,len(mesh)))
        data_gen_Phi = np.zeros((2,len(mesh)))
        data_gen_Fi_t = np.zeros((2,len(mesh)))
        data_gen_ms_birth = np.zeros(len(mesh))

        # BEGIN: FOR EACH PARTICLE ###########################################
        for _ in range(num_particles):
            n_exist = True
            n_E = 1 # n's always born fast
            m, x = fn_det_pos_birth(mesh, mesh_fuel)
            data_gen_ms_birth[m] += 1
            mu = 2*np.random.rand() - 1
            if mu == 0: # just to be safe
                mu_decider = np.random.rand()
                if mu_decider < 0.5:
                    mu -= 0.001 
                else:
                    mu += 0.001
            
            # BEGIN: WHILE PARTICLE EXISTS ###################################
            while n_exist:   

                S = -np.log(np.random.random())/mesh[m][f'Sigma_t_{n_E}']
                travel_dist = mu*S # is called Delta_x in handout pseudocode
                crosses = 0
                
                # BEGIN: WHILE PARTICLE REMAINS IN SAME MATERIAL #############
                while crosses == 0 and n_exist:
                    x_new = x + travel_dist

                    # BEGIN: IF PARTICLE MOVES TO THE LEFT ###################
                    if mu < 0:
                        locs_left_m = mesh[m]['locs_left']
                        if x_new < locs_left_m:
                            data_gen_TL[n_E-1][m] += abs((x - locs_left_m) / mu)
                            data_gen_J[n_E-1][m] += mu
                            travel_dist = travel_dist + (x - mesh[m]['locs_left'])
                            x = locs_left_m
                            m_old = m
                            m -= 1
                            
                            if m < 0:
                                n_exist, m, mu, travel_dist = fn_det_BC_interaction(input_file.bc_1,
                                                                                    m, m_old, mu, travel_dist)
                            else:
                                mat_id_m = mesh[m]['mat_ID']
                                mat_id_m_old = mesh[m_old]['mat_ID']
                                if mat_id_m != mat_id_m_old:
                                    crosses = 1
                        else:
                            data_gen_TL[n_E-1][m] += abs(travel_dist / mu)
                            data_gen_J[n_E-1][m] += mu
                            x = x +travel_dist
                            n_exist, n_E, mu, S, travel_dist = fn_det_mesh_interaction(mesh, m, n_E, mu, S, travel_dist)
                    # END:   IF PARTICLE MOVES TO THE LEFT ###################



                    # BEGIN: IF PARTICLE MOVES TO THE RIGHT ##################
                    else:
                        if x_new > mesh[m]['locs_right']:
                            locs_right_m = mesh[m]['locs_right']
                            data_gen_TL[n_E-1][m] += abs((locs_right_m - x) / mu)
                            data_gen_J[n_E-1][m] += mu
                            travel_dist = travel_dist - (mesh[m]['locs_right'] - x)
                            x = locs_right_m
                            m_old = m
                            m += 1

                            if m > (len(mesh) - 1):
                                n_exist, m, mu, travel_dist = fn_det_BC_interaction(input_file.bc_2,
                                                                                    m, m_old, mu, travel_dist)
                            else:
                                mat_id_m = mesh[m]['mat_ID']
                                mat_id_m_old = mesh[m_old]['mat_ID']
                                if mat_id_m != mat_id_m_old:
                                    crosses = 1
                        else:
                            data_gen_TL[n_E-1][m] += abs(travel_dist / mu)
                            data_gen_J[n_E-1][m] += mu
                            x = x + travel_dist
                            n_exist, n_E, mu, S, travel_dist = fn_det_mesh_interaction(mesh, m, n_E, mu, S, travel_dist)

                    # END:   IF PARTICLE MOVES TO THE RIGHT ##################
                # END:   WHILE PARTICLE REMAINS IN SAME MATERIAL #############  
            # END:   WHILE PARTICLE EXISTS ###################################
        # END:   FOR EACH PARTICLE ###########################################



        # BEGIN: CALCULATE PARAMETERS OF INTEREST ############################
        if (g+1) > (0.1*input_file.num_gen):
            Delta_x_list = np.array([mesh['Delta_x'] for mesh in mesh if 'Delta_x' in mesh])
            data_gen_Phi = data_gen_TL/(k*num_particles*Delta_x_list)

            v_f_1_list = np.array([mesh['v_f_1'] for mesh in mesh if 'v_f_1' in mesh])
            Sigma_f_1_list = np.array([mesh['Sigma_f_1'] for mesh in mesh if 'Sigma_f_1' in mesh])
            n_E_1_list = data_gen_Phi[0]
            data_gen_Fi_1 = v_f_1_list*Sigma_f_1_list*n_E_1_list

            v_f_2_list = np.array([mesh['v_f_2'] for mesh in mesh if 'v_f_2' in mesh])
            Sigma_f_2_list = np.array([mesh['Sigma_f_2'] for mesh in mesh if 'Sigma_f_2' in mesh])
            n_E_2_list = data_gen_Phi[1]
            data_gen_Fi_2 = v_f_2_list*Sigma_f_2_list*n_E_2_list
            
            data_gen_Fi_t = np.transpose(data_gen_Fi_1 + data_gen_Fi_2)
            k = k * (Delta_x_list*data_gen_Fi_t).sum().sum()
            ks.append(k)
            print(f'k: {k}')

            data_tot_TL = np.concatenate((data_tot_TL, data_gen_TL))
            data_tot_J = np.concatenate((data_tot_J, data_gen_J))
            data_tot_Phi = np.concatenate((data_tot_Phi, data_gen_Phi))
            data_tot_Fi_t = np.concatenate((data_tot_Fi_t, data_gen_Fi_t)) 
            data_tot_ms_birth = np.concatenate((data_tot_ms_birth, data_gen_ms_birth)) 
        # END:   CALCULATE PARAMETERS OF INTEREST ############################
    # END:   FOR EACH GENERATION #############################################


    # BEGIN: CALCULATE FUNDAMENTAL MODES #####################################

    TL_fund_1 = (np.sum(data_tot_TL[::2, :], axis=0, keepdims=True)/num_gen).T
    TL_fund_2 = (np.sum(data_tot_TL[1::2, :], axis=0, keepdims=True)/num_gen).T
    J_fund_1 = (np.sum(data_tot_J[::2, :], axis=0, keepdims=True)/num_gen).T
    J_fund_2 = (np.sum(data_tot_J[1::2, :], axis=0, keepdims=True)/num_gen).T
    Phi_fund_1 = (np.sum(data_tot_Phi[::2, :], axis=0, keepdims=True)/num_gen).T
    Phi_fund_2 = (np.sum(data_tot_Phi[1::2, :], axis=0, keepdims=True)/num_gen).T
    k_fund = sum(ks)/num_gen
    print(f'k_fund = {k_fund}')
    # else:
    #     ... = pd.DataFrame(0, index=['n_E=1', 'n_E=2'], columns=mesh.columns)
    data = np.concatenate((TL_fund_1, TL_fund_2,
                           J_fund_1, J_fund_2,
                           Phi_fund_1, Phi_fund_2), axis=1).T
    
    data_ks = pd.DataFrame({'ks':ks, 'k_fund':k_fund})
    data_tot_ms_birth = pd.DataFrame({data_tot_ms_birth.sum(axis = 0)})
    # END:   CALCULATE FUNDAMENTAL MODES #####################################


    # BEGIN: OUTPUTS #########################################################
    timestr = time.strftime('%Y%m%d_%H%M%S')

    filename = f'MultiAssyFlux_results_MC_g{num_gen}_n{num_particles}_{timestr}.csv'
    fp_data = os.path.join(dir_output, filename)
    index_names = ['TL_fund_1','TL_fund_2',
                   'J_fund_1', 'J_fund_2',
                   'Phi_fund_1', 'Phi_fund_2']
    data = np.vstack((index_names, data.T))
    np.savetxt(filename, data, delimiter=',', fmt='%s')

    filename_ks = f'MultiAssyFlux_ks_MC_g{num_gen}_n{num_particles}_{timestr}.csv'
    fp_ks = os.path.join(dir_output, filename_ks)
    data_ks.to_csv(fp_ks)

    filename_birth = f'MultiAssyFlux_loc_births_MC_g{num_gen}_n{num_particles}_{timestr}.csv'
    fp_birth = os.path.join(dir_output, filename_birth)
    data_tot_ms_birth.to_csv(fp_birth)
    
    time_elapsed_solnMC = time.time() - time_start_solnMC
    time_p = (time_elapsed_solnMC/num_gen/num_particles)*1000 # milliseconds
    print(f"""
##########################################################################
Monte Carlo solution complete.
Results exported to {filename}
in \{dir_output}\

- {num_gen} generations.
- {num_particles} particles/generation.
- Run time: {time_elapsed_solnMC:.3f} s = {time_elapsed_solnMC/60:.3f} m = {time_elapsed_solnMC/3600:.3f} h
- Time per particle: {time_p:0.3f} ms.

- Final k: {k}
- k_fund: {k_fund}
##########################################################################
    """)
    # END:   OUTPUTS #########################################################

    return data, data_ks, fp_data, fp_ks
# END:   MONTE CARLO SOLUTION FUNCTION #######################################

