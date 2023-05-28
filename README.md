# MultiAssyFlux README  
Multi-assembly, 1-D, pin-by-pin Monte Carlo flux solution code

## Introduction
MultiAssyflux calculates neutron flux, current, and multiplication factor for 1-D assemblies consisting of notional Uranium or MOX fuels, the only distinction being cross-section values. Pure reflector, vacuum, and water reflector (implemented by adding a full assembly consisting purely of water to each end specified) boundar conditions are possible. 

The fundamental principles of this work can be found in: Nuclear Reactor Analysis, J.J. Duderstadt and L.J. Hamilton, 1976, (ISBN: 978-0-471-22363-4).

## Getting started
### Requirements
See also requirements.txt  
- numpy
- matplotlib
- pandas

### Inputs
Input files must be python files. All variables within the provided input file (/inputs/MC_1.py) must be defined. The values included are reasonable. Cross section set A (x_sections_set_A.csv) is a simplied set best for debugging. Set B provides more realistic values.

### Outputs
Results will be exported to ./outputs/ into a new folder for each run. Data csv files and useful plots are created, including a Bokeh interactive plot in the form of an html file that will auto-open upon sucessful run completion

### Useage
#### If within current directory
```
python MultiAssyFlux_Setup.py -i <./local/path/to/input/filename>.py  
```
#### If not
```
python MultiAssyFlux_Setup.py -i </entire/path/to/input/filename>.py  
```

### Known issues and limitations
This is a 1-D code. It solves the problems given using basic Monte Carlo transport/tracking principles given only a 2-group (far from realistic) cross section dataset.

## Version history
- v1.0.0: initial release, 230515 BAT

## License
GNU GPL v3.0

## Authors
Brice Turner

## Acknowledgements
This work would not be possible without the teachings of Justin C. Watson, PhD, of the University of Florida.

