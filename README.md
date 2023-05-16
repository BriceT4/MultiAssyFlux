# MultiAssyFlux README  
Multi-assembly, 1-D, pin-by-pin Monte Carlo flux solution code

## Introduction
MultiAssyflux calculates neutron flux, current, and multiplication factor for 1-D assemblies consisting of notional Uranium or MOX fuels, the only distinction being cross-section values. Pure reflector, vacuum, and water reflector (implemented by adding a full assembly consisting purely of water to each end specified) boundar conditions are possible. 

The fundamental principles of this work can be found in: Nuclear Reactor Analysis, J.J. Duderstadt and L.J. Hamilton, 1976, (ISBN: 978-0-471-22363-4).
## Getting started

### Requirements

### Inputs

### Outputs

### Useage
#### If within current directory
```
python MultiAssyFlux_Setup.py -i <.\path\to\input\filename>.py  
```
#### If not
```
python MultiAssyFlux_Setup.py -i <\entire\path\to\input\filename>.py  
```

### Known issues and limitations

## Version history

## License
GNU GPL v3.0

## Authors
Brice Turner

## Acknowledgements
This work would not be possible without the teachings of Justin C. Watson, PhD, of the University of Florida.

