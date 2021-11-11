# EV-CSL
Electric Vehicle Charging Stations Location Problem

## Setup

```console
source ./venv/bin/activate
python3 -m pip install -r requirements.txt -f libs
```

### Generate libraries

```console
pip freeze --local > requirements.txt
pip download -r ../requirements.txt --no-index --find-links `pwd`
```

## Run

```console
python3 main.py <Fixed Parameters> <Opional Parameters>
```
Fixed Parameters:
* _[Number clients]_
* _[Number locations]_
* _[Number electrical stations]_
* _[Matrix clients x locations]_
* _[Matrix electrical stations x locations]_
* _[Clients data file]_
* _[Locations data file]_
* _[Electrical substation data file]_

Optional Parameters:
* --seed _[Random seed]_ 
* --algo _[Algorithm]_: Options [NSGA2,SPEA2]
* --pop _[population size]_
* --prob-cross _[Crossover probability]_
* --prob-mut _[Mutation probability]_
* --ga-sel _[Selection operator]_: Options [TOURNAMENT2]
* --ga-cross _[Crossover operator]_: Options [2POINT]
* --ga-mut _[Mutation operator]_: Options [UNIFORM]
* --ga-repl _[Replacement operator]_: Options [mu_lambda]. If the algorithm is NSGAII or SPEA2 this option is fixed.
* --iter _[Maximun number of iterations]_
* --radius _[Radius size]_: in meters
* --qos-alpha _[number]_: weight to the qos term in the QoS fitness function
* --qos-beta _[number]_: weight to the overlapping term in the QoS fitness function
* --qos-gamma _[number: weight to the not service term in the QoS fitness function]_
* --input _[folder]_: Folder where the input data is
* --output _[ folder]_: Folder where write the output files
* ./outputs


### Example
```console
venv/bin/python3.8 main.py 363 33550 14 CxA_filter.ssv ExA.ssv cp.ssv facility_points.ssv substations_points_EkW.ssv --pop 20 --ga-sel None --prob-cross 0.7 --prob-mut 0.2 --ga-sel TOURNAMENT2 --ga-cross 2POINT --ga-mut UNIFORM --ga-repl mu_lambda --pop 20 --iter 1000 --radius 500 --input ./inputs --output ./outputs
```
