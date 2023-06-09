#!/usr/bin/python
# genMesh.py
# A MultiAssyFlux module
# (C) Brice Turner, 2023

import numpy as np
import os
import time
import csv

time_start_genMesh = time.time()


def genMesh(input_file):
    num_mesh = input_file.num_mesh
    thick_cell_fuel = input_file.D
    thick_cell_mod = input_file.Pitch - thick_cell_fuel
    thick_cell_mod_ends = thick_cell_mod / 2
    Delta_x_fuel = thick_cell_fuel / num_mesh
    Delta_x_mod = thick_cell_mod / num_mesh
    Delta_x_mod_ends = thick_cell_mod_ends / num_mesh
    num_gen = input_file.num_gen
    num_particles = input_file.num_particles

    mat_U = input_file.mat_U
    mat_MOX = input_file.mat_MOX
    mat_water_ref = input_file.mat_water_ref

    mesh_mat_1 = mat_U if input_file.mat_1 == 'Uranium' else mat_MOX
    mesh_mat_2 = mat_U if input_file.mat_2 == 'Uranium' else mat_MOX

    if input_file.bc_1 == 'Water reflector' and input_file.bc_2 == 'Water reflector':
        mesh_mat_tot = np.concatenate((mat_water_ref, mesh_mat_1, mesh_mat_2, mat_water_ref))
    elif input_file.bc_1 == 'Water reflector':
        mesh_mat_tot = np.concatenate((mat_water_ref, mesh_mat_1, mesh_mat_2))
    elif input_file.bc_2 == 'Water reflector':
        mesh_mat_tot = np.concatenate((mesh_mat_1, mesh_mat_2, mat_water_ref))
    else:
        mesh_mat_tot = np.concatenate((mesh_mat_1, mesh_mat_2))

    mesh_mat = np.repeat(mesh_mat_tot, num_mesh)
    mesh = [{'mat_ID': i} for i in mesh_mat.tolist()]

    material_map = {
        0: 'water',
        1: 'U_fuel',
        2: 'MOX',
        3: 'control_rod',
        4: 'air'
    }

    mesh = [{'mat_ID': row['mat_ID'], 'material': material_map[row['mat_ID']]} for row in mesh]

    # Load x_sections_all data and headers
    x_sections_all = {}
    with open(input_file.x_sections_set, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            x_sections_all[row[0]] = np.array([float(value) for value in row[1:]])

    for i, row in enumerate(mesh):
        m = row['mat_ID']
        x_secs = x_sections_all.get(material_map[m])
        if x_secs is not None:
            for idx, x_sec in enumerate(x_secs):
                column_name = headers[idx + 1]
                row[column_name] = x_sec

    # Add x positions to mesh data
    loc_current = 0
    for i, row in enumerate(mesh):
        m = row['mat_ID']
        if i < num_mesh:
            Delta_x = Delta_x_mod_ends
            loc_left = loc_current
            loc_center = loc_left + 0.5 * Delta_x_mod_ends
            loc_right = loc_center + 0.5 * Delta_x_mod_ends
        elif i < (len(mesh) // 2 - num_mesh):
            Delta_x = Delta_x_mod if m == 0 else Delta_x_fuel
            loc_left = loc_current
            loc_center = loc_left + 0.5 * Delta_x
            loc_right = loc_center + 0.5 * Delta_x
        elif i < (len(mesh) // 2 + num_mesh):
            Delta_x = Delta_x_mod_ends
            loc_left = loc_current
            loc_center = loc_left + 0.5 * Delta_x_mod_ends
            loc_right = loc_center + 0.5 * Delta_x_mod_ends
        elif i < (len(mesh) - num_mesh):
            Delta_x = Delta_x_mod if m == 0 else Delta_x_fuel
            loc_left = loc_current
            loc_center = loc_left + 0.5 * Delta_x
            loc_right = loc_center + 0.5 * Delta_x
        else:
            Delta_x = Delta_x_mod_ends
            loc_left = loc_current
            loc_center = loc_left + 0.5 * Delta_x_mod_ends
            loc_right = loc_center + 0.5 * Delta_x_mod_ends

        loc_current = loc_right

        row.update({
            'locs_left': loc_left,
            'locs_center': loc_center,
            'locs_right': loc_right,
            'Delta_x': Delta_x
        })

    mesh_fuel = [row for _, row in enumerate(mesh) if row['mat_ID'] != 0 and row['mat_ID'] != 3]
    cols_old = [index for index, row in enumerate(mesh) if row['mat_ID'] != 0 and row['mat_ID'] != 3]

    for dictionary, col in zip(mesh_fuel, cols_old):
        dictionary['cols_old'] = col

    timestr = time.strftime('%Y%m%d_%H%M%S')
    filename = f'MultiAssyFlux_mesh_{timestr}.csv'
    dir_output = f'outputs/MultiAssyFlux_outputs_MC_g{num_gen}_n{num_particles}_{timestr}'
    os.makedirs(dir_output, exist_ok=True)
    fp = os.path.join(dir_output, filename)

    with open(fp, 'w', newline='') as csvfile:
        fieldnames = list(mesh[0].keys()) + ['cols_old']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mesh)

    time_elapsed_genMesh = time.time() - time_start_genMesh

    print(f"""
##########################################################################
Mesh generation complete.
Mesh exported to {filename}
in {dir_output}
Run time: {time_elapsed_genMesh:.3f} s.
##########################################################################
""")

    return mesh, mesh_fuel, dir_output