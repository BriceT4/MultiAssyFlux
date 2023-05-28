#!/usr/bin/python
# MultiAssyFlux_Setup.py
# Base of MultiAssyFlux project
# UFL ENU6106 Spring 2023 Term Project
# Brice Turner, 2023

import argparse
import importlib
import os

# BEGIN: COMMAND LINE INTERFACE ##############################################
def setupCommandLine():
    print(f"""
##############################################################################
       __  __       _ _   _                       _______ _            
      |  \/  |     | | | (_)   /\                |   ____| |           
      | \  / |_   _| | |_ _   /  \  ___ ___ _   _|  |__  | |_   ___  __
      | |\/| | | | | | __| | / /\ \/ __/ __| | | |   __| | | | | \ \/ /
      | |  | | |_| | | |_| |/ ____ \__ \__ \ |_| |  |    | | |_| |>  < 
      |_|  |_|\__,_|_|\__|_/_/    \_\___/___\___ |__|    |_|\__,_/_/\_\         
                                             __/ |                               
                                            |___/                                       
############################################################################## 
    """)

    desc = f'Command line interface for {os.path.basename(__file__)}.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-i', '--INPUT', nargs='?', dest='infile',
                        metavar='input_file', required=True,
                        help='Input file name. REQUIRED. Must be .py.\n'
                            'Usage: -i/--INPUT <filename>.py')
    args = parser.parse_args()

    # Get the file path and the file name without the extension
    file_path = os.path.abspath(args.infile)
    infile_name = os.path.splitext(os.path.basename(args.infile))[0]

    # Replace periods/dots with underscores in the module name
    infile_name = infile_name.replace('.', '_')

    # Make sure the input file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{args.infile}' not found")

    # Import the input file as a module
    spec = importlib.util.spec_from_file_location(infile_name, file_path)
    input_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(input_file)

    return input_file
# END:   COMMAND LINE INTERFACE ##############################################






# BEGIN: SETUP ###############################################################
def Setup(InputFile):
    from src.genMesh import genMesh
    mesh, mesh_fuel, dir_output = genMesh(InputFile)

    from src.solnMC import solnMC
    data, data_ks, fp_data, fp_ks = solnMC(InputFile, mesh, mesh_fuel, dir_output)

    return mesh, mesh_fuel, dir_output, data, data_ks, fp_data, fp_ks
# END:   SETUP ###############################################################


# BEGIN: PLOT ################################################################
def plotIt(input_file, mesh, data_ks, fp_data, fp_ks, dir_output):
    from src.genPlots import plotter
    plotter(input_file, mesh, data_ks, fp_data, fp_ks, dir_output)
    return
# END:   PLOT ################################################################


def printSuccess():
    print(f"""
##############################################################################
{os.path.basename(__file__)} sucessfully ran.
##############################################################################
    """)

input_file = setupCommandLine()
mesh, mesh_fuel, dir_output, data, data_ks, fp_data, fp_ks = Setup(input_file)
plotIt(input_file, mesh, data_ks, fp_data, fp_ks, dir_output)
printSuccess()

