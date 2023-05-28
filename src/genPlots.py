#!/usr/bin/python
# solnMC.py
# A MultiAssyFlux module
# (C) Brice Turner, 2023


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time

from bokeh.plotting import figure
from bokeh.io import output_file, show
from bokeh.models import HoverTool

# BEGIN: DEFINE FUNCTION #####################################################
def plotter(input_file, mesh, data_ks, fp_data, fp_ks, dir_output):
    data_my = pd.read_csv(fp_data)
    data_ks = pd.read_csv(fp_ks, header = None)

    mesh = pd.DataFrame(mesh)

    ms = mesh['locs_center']
    mats = mesh['material']
    TL_1 = data_my['TL_fund_1']
    TL_2 = data_my['TL_fund_2']
    J_fund_1 = data_my['J_fund_1']
    J_fund_2 = data_my['J_fund_2']
    Phi_fund_1 = data_my['Phi_fund_1']
    Phi_fund_2 = data_my['Phi_fund_2']

    ks = data_ks.iloc[:,0]
    ks = pd.concat([pd.Series([1]), ks]).reset_index(drop=True)
    

    # BEGIN: STATIC FLUX PLOT ################################################
    plt.plot(ms, Phi_fund_1, color = 'k', label = '$\Phi_{fund,1}$') # , drawstyle='steps-post'
    plt.plot(ms, Phi_fund_2, color = 'r', label = '$\Phi_{fund,2}$') # , drawstyle='steps-post'

    for idx, material in enumerate(mats):
        if idx == 0:
            continue
        if material == 'water':
            color = 'blue'
        else:
            color = 'grey'
        x_fill = np.linspace(ms[idx-1], ms[idx], 100)
        plt.fill_between(x_fill, -10, 10, color=color, alpha=0.2)

    plt.xlabel('Position (cm)')
    plt.ylim([0, 1.5])
    # plt.semilogy()
    plt.legend()
    plt.tight_layout()

    timestr = time.strftime('%Y%m%d_%H%M%S')
    filename = f'MultiAssyFlux_plots_flux_{timestr}.png'
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    fp_plot_static = os.path.join(dir_output, filename)
    fig = plt.gcf()
    fig.savefig(fp_plot_static, dpi=600)
    plt.show()
    # END:   STATIC FLUX PLOT ################################################


    # BEGIN: STATIC K PLOT ###################################################
    x = range(1, input_file.num_gen+1)
    plt.plot(x, ks, drawstyle = 'steps-post')
    plt.xlabel('Generation')
    plt.ylabel('Multiplication factor, $k$ (arb. unit)')
    plt.xticks(np.arange(min(x), max(x)+1, 1))

    plt.ylim([0, 1.5])
    # plt.semilogy()
    plt.tight_layout()

    timestr = time.strftime('%Y%m%d_%H%M%S')
    filename_k = f'MultiAssyFlux_plots_k_{timestr}.png'
    if not os.path.exists(dir_output):
        os.makedirs(dir_output)
    fp_plot_k = os.path.join(dir_output, filename_k)
    fig_k = plt.gcf()
    fig_k.savefig(fp_plot_k, dpi=600)
    plt.show()
    # END:   STATIC K PLOT ###################################################



    # BEGIN: BOKEH PLOT ######################################################
    # create a new plot with figure
    myFontSize = 10
    p = figure(title = f'MultiAssyFlux_plotsInteractive_{timestr}.html', width = 1800, height = 750, y_axis_type="log") #
    # p.title.text_font_size = myFontSize

    p.line(ms, TL_1, line_color = 'blue', legend_label = 'TL_1')
    p.line(ms, TL_2, line_color = 'orange', legend_label = 'TL_2')
    p.line(ms, J_fund_1, line_color = 'black', legend_label = 'J_fund_1')
    p.line(ms, J_fund_2, line_color = 'orange', legend_label = 'J_fund_2')
    p.line(ms, Phi_fund_1, line_color = 'black', legend_label = 'Phi_fund_1')
    p.line(ms, Phi_fund_2, line_color = 'red', legend_label = 'Phi_fund_2')

    p.add_tools(HoverTool(
        tooltips = [
            ('y', '@y'),
            ('x', '@x')
            ],
        mode = 'vline'
        ))

    p.xaxis.axis_label = 'Position (cm)'
    # p.yaxis.axis_label = 'Temperature Rise and dT/dt [K, K/s]'
    # p.axis.axis_label_text_font_style = 'bold'
    # p.axis.axis_label_text_font_size = myFontSize

    p.legend.click_policy="hide"
    # p.legend.label_text_font_size = myFontSize

    # This line puts the legend outside of the plot area
    p.add_layout(p.legend[0], 'right')

    output_file(f'{dir_output}\\MultiAssyFlux_plotsInteractive_{timestr}.html',
                title = f'MultiAssyFlux_plotsInteractive_{timestr}.html')
    show(p)
    # output_file(output_path)

    # p.savefig(output_file)
    # save(p)
    # show(p)
    # END: BOKEH PLOT   ######################################################

    # plt.show()
    print(f"""
##########################################################################
Plotting complete.
Plots exported to \{dir_output}\ 
##########################################################################
    """)
    return 
# END:   DEFINE FUNCTION #####################################################