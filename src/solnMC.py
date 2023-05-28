#!/usr/bin/python
# solnMC.py
# A MultiAssyFlux module
# (C) Brice Turner, 2023

import numpy as np
np.random.seed(0)
import os
import time


# BEGIN: MONTE CARLO SOLUTION FUNCTION #######################################
time_start_solnMC = time.time()
def solnMC(input_file, mesh, mesh_fuel, dir_output):

    size_mesh_fuel = mesh_fuel.shape[1] - 1

    # BEGIN SAMPLE PARTICLE BIRTH POSITION ###################################
    def fn_det_pos_birth(mesh, mesh_fuel):
        loc_rand_mesh_fuel = np.random.randint(0, size_mesh_fuel)
        m = mesh_fuel['cols_old', loc_rand_mesh_fuel]
        mesh_vals = mesh[['locs_left', 'Delta_x'], m]
        r = np.random.uniform()
        x = mesh_vals['locs_left'] + r * mesh_vals['Delta_x']

        return m, x
    # END:  SAMPLE PARTICLE BIRTH POSITION ###################################



    # BEGIN: DETERMINE INTERACTION TYPE ######################################
    def fn_det_mesh_interaction(mesh, m, n_E, mu, S, travel_dist):
        R = np.random.rand()

        Sigma_t = mesh[f'Sigma_t_{n_E}', m]
        Sigma_a = mesh[f'Sigma_a_{n_E}', m]
        Sigma_f = mesh[f'Sigma_f_{n_E}', m]
        Sigma_inscatter = mesh[f'Sigma_s-0_{n_E}-{n_E}', m]

        lim_upper_capture = (Sigma_a - Sigma_f) / Sigma_t
        lim_upper_fission = Sigma_a / Sigma_t
        lim_upper_in_scatter = (Sigma_inscatter + Sigma_a) / Sigma_t

        if R < lim_upper_capture:
            n_exist = False
        elif lim_upper_capture <= R < lim_upper_fission:
            n_exist = False
        elif lim_upper_fission <= R < lim_upper_in_scatter:
            n_exist = True
            mu = 2 * np.random.rand() - 1
            S = -np.log(np.random.random()) / Sigma_t
            travel_dist = mu * S
        elif lim_upper_in_scatter <= R:
            n_exist = True
            n_E = 2
            mu = 2 * np.random.rand() - 1
            S = -np.log(np.random.random()) / mesh[f'Sigma_t_2', m]
            travel_dist = mu * S
        else:
            n_exist = False

        return n_exist, n_E, mu, S, travel_dist
    # END:   DETERMINE INTERACTION TYPE ######################################
    
    
    
    # BEGIN: DETERMINE BOUNDARY INTERACTION ##################################
    def fn_det_BC_interaction(BC, m, m_old, mu, travel_dist):
        if BC == 'Pure reflector' or BC == 'Water reflector':
            n_exist = True
            mu = -1 * mu
            travel_dist = -1 * travel_dist
            m = m_old
        else:
            n_exist = False

        return n_exist, m, mu, travel_dist
    # END:   DETERMINE BOUNDARY INTERACTION ##################################



    # BEGIN: INITIALIZE VARIABLES ############################################
    data = np.zeros((6, mesh.shape[1]))
    data_tot_J = np.zeros_like(mesh)
    data_tot_TL = np.zeros_like(mesh)
    data_tot_Phi = np.zeros_like(mesh)
    data_tot_Fi_t = np.zeros_like(mesh)
    data_tot_ms_birth = np.zeros(mesh.shape[1])
    num_gen = input_file.num_gen
    num_particles = input_file.num_particles
    ks = []
    k = 1
    # END: INITIALIZE VARIABLES ############################################


    # BEGIN: FOR EACH GENERATION #############################################
    for g in range(num_gen):
        print(f'\t######## Generation: {g+1} ########')

        data_gen_TL = np.zeros_like(mesh)
        data_gen_J = np.zeros_like(mesh)
        data_gen_Phi = np.zeros_like(mesh)
        data_gen_Fi_t = np.zeros_like(mesh)
        data_gen_ms_birth = np.zeros(mesh.shape[1])

        # BEGIN: FOR EACH PARTICLE ###########################################
        for _ in range(num_particles):
            n_exist = True
            n_E = 1
            m, x = fn_det_pos_birth(mesh, mesh_fuel)
            data_gen_ms_birth[m] += 1
            mu = 2 * np.random.rand() - 1
            if mu == 0:
                mu_decider = np.random.rand()
                if mu_decider < 0.5:
                    mu -= 0.001
                else:
                    mu += 0.001
            
            # BEGIN: WHILE PARTICLE EXISTS ###################################
            while n_exist:   

                S = -np.log(np.random.random()) / mesh[f'Sigma_t_{n_E}', m]
                travel_dist = mu * S
                crosses = 0
                
                # BEGIN: WHILE PARTICLE REMAINS IN SAME MATERIAL #############
                while crosses == 0 and n_exist:
                    x_new = x + travel_dist

                    # BEGIN: IF PARTICLE MOVES TO THE LEFT ###################
                    if mu < 0:
                        locs_left_m = mesh['locs_left', m]
                        if x_new < locs_left_m:
                            data_gen_TL[f'n_E={n_E}', m] += abs((x - locs_left_m) / mu)
                            data_gen_J[f'n_E={n_E}', m] += mu
                            travel_dist = travel_dist + (x - mesh['locs_left', m])
                            x = locs_left_m
                            m_old = m
                            m -= 1
                            
                            if m < 0:
                                n_exist, m, mu, travel_dist = fn_det_BC_interaction(input_file.bc_1,
                                                                                    m, m_old, mu, travel_dist)
                            else:
                                mat_id_m = mesh['mat_ID', m]
                                mat_id_m_old = mesh['mat_ID', m_old]
                                if mat_id_m != mat_id_m_old:
                                    crosses = 1
                        else:
                            data_gen_TL[f'n_E={n_E}', m] += abs(travel_dist / mu)
                            data_gen_J[f'n_E={n_E}', m] += mu
                            x = x + travel_dist
                            n_exist, n_E, mu, S, travel_dist = fn_det_mesh_interaction(mesh, m, n_E, mu, S, travel_dist)
                    # END:   IF PARTICLE MOVES TO THE LEFT ###################



                    # BEGIN: IF PARTICLE MOVES TO THE RIGHT ##################
                    else:
                        if x_new > mesh['locs_right', m]:
                            locs_right_m = mesh['locs_right', m]
                            data_gen_TL[f'n_E={n_E}', m] += abs((locs_right_m - x) / mu)
                            data_gen_J[f'n_E={n_E}', m] += mu
                            travel_dist = travel_dist - (mesh['locs_right', m] - x)
                            x = locs_right_m
                            m_old = m
                            m += 1

                            if m > (mesh.shape[1] - 1):
                                n_exist, m, mu, travel_dist = fn_det_BC_interaction(input_file.bc_2,
                                                                                    m, m_old, mu, travel_dist)
                            else:
                                mat_id_m = mesh['mat_ID', m]
                                mat_id_m_old = mesh['mat_ID', m_old]
                                if mat_id_m != mat_id_m_old:
                                    crosses = 1
                        else:
                            data_gen_TL[f'n_E={n_E}', m] += abs(travel_dist / mu)
                            data_gen_J[f'n_E={n_E}', m] += mu
                            x = x + travel_dist
                            n_exist, n_E, mu, S, travel_dist = fn_det_mesh_interaction(mesh, m, n_E, mu, S, travel_dist)

                    # END:   IF PARTICLE MOVES TO THE RIGHT ##################
                # END:   WHILE PARTICLE REMAINS IN SAME MATERIAL #############  
            # END:   WHILE PARTICLE EXISTS ###################################
        # END:   FOR EACH PARTICLE ###########################################



        # BEGIN: CALCULATE PARAMETERS OF INTEREST ############################
        if (g+1) > (0.1*num_gen):
            data_gen_Phi = data_gen_TL / (k * num_particles * mesh['Delta_x', :])
            data_gen_Fi_1 = mesh['v_f_1', :] * mesh['Sigma_f_1', :] * data_gen_Phi[f'n_E=1', :]
            data_gen_Fi_2 = mesh['v_f_2', :] * mesh['Sigma_f_2', :] * data_gen_Phi[f'n_E=2', :]
            data_gen_Fi_t = data_gen_Fi_1 + data_gen_Fi_2
            k = k * (mesh['Delta_x', :] * data_gen_Fi_t).sum()
            ks.append(k)
            print(f'k: {k}')

            data_tot_TL += data_gen_TL
            data_tot_J += data_gen_J
            data_tot_Phi += data_gen_Phi
            data_tot_Fi_t += data_gen_Fi_t    
            data_tot_ms_birth += data_gen_ms_birth    
        # END:   CALCULATE PARAMETERS OF INTEREST ############################
    # END:   FOR EACH GENERATION #############################################


    # BEGIN: CALCULATE FUNDAMENTAL MODES #####################################
    TL_fund_1 = (data_tot_TL[f'n_E=1', :].sum(axis=0) / num_gen).transpose()
    TL_fund_2 = (data_tot_TL[f'n_E=2', :].sum(axis=0) / num_gen).transpose()
    J_fund_1 = (data_tot_J[f'n_E=1', :].sum(axis=0) / num_gen).transpose()
    J_fund_2 = (data_tot_J[f'n_E=2', :].sum(axis=0) / num_gen).transpose()
    Phi_fund_1 = (data_tot_Phi[f'n_E=1', :].sum(axis=0) / num_gen).transpose()
    Phi_fund_2 = (data_tot_Phi[f'n_E=2', :].sum(axis=0) / num_gen).transpose()
    k_fund = sum(ks) / num_gen
    print(f'k_fund = {k_fund}')
    data = np.concatenate((TL_fund_1, TL_fund_2,
                        J_fund_1, J_fund_2,
                        Phi_fund_1, Phi_fund_2)).reshape(6, -1)

    data_ks = np.array(ks).reshape(-1, 1)
    data_tot_ms_birth = data_tot_ms_birth.sum(axis=0)
    # END:   CALCULATE FUNDAMENTAL MODES #####################################


    # BEGIN: OUTPUTS #########################################################
    timestr = time.strftime('%Y%m%d_%H%M%S')

    filename = f'MultiAssyFlux_results_MC_g{num_gen}_n{num_particles}_{timestr}.csv'
    fp_data = os.path.join(dir_output, filename)
    index_names = ['TL_fund_1', 'TL_fund_2',
                'J_fund_1', 'J_fund_2',
                'Phi_fund_1', 'Phi_fund_2']
    np.savetxt(fp_data, data, delimiter=',', header=','.join(index_names))

    filename_ks = f'MultiAssyFlux_ks_MC_g{num_gen}_n{num_particles}_{timestr}.csv'
    fp_ks = os.path.join(dir_output, filename_ks)
    np.savetxt(fp_ks, data_ks, delimiter=',')

    filename_birth = f'MultiAssyFlux_loc_births_MC_g{num_gen}_n{num_particles}_{timestr}.csv'
    fp_birth = os.path.join(dir_output, filename_birth)
    np.savetxt(fp_birth, data_tot_ms_birth, delimiter=',')

    time_elapsed_solnMC = time.time() - time_start_solnMC
    time_p = (time_elapsed_solnMC / (num_gen * num_particles)) * 1000  # milliseconds
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
