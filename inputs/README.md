# Inputs file description

All the files have space-separated-values (ssv) format.

## Distance matrix between clients and possible locations

The file *CxA_filter.ssv* is a float matrix of Customers x Locations (CxL).
Without headers or row indexes.
The values are the street walking distance in meters.

Example
   | l1  | l2  | l3  | l4   |
c1 | 2.1 | 4.6 | 8.2 | 1.1  |
c2 | 6.2 | 1.8 | 5.1 | 0.2  |
c3 | 1.3 | 2.7 | 7.1 | 10.9 |

## Distance matrix between energy stations (energy supplies) and possible locations
The file *pertinencia.ssv* is a binary matrix of Energy-stations x Locations (ExL)
Without headers or row indexes.
A 1 means that the location _l_ is in the range of supply of the station _e_.

Example
   | l1 | l2 | l3 | l4 | l5 |
c1 | 1  | 0  | 0  | 0  | 0  |
c2 | 0  | 1  | 1  | 0  | 0  |
c3 | 0  | 0  | 0  | 0  | 1  |
c4 | 0  | 0  | 0  | 1  | 0  |

## Clients data
The file *cp.ssv* contains the data of each client (row).
Each column contains the information about:
- latitude
- longitude
- number of people that live in this locations

## Location data
The file *facility_points.ssv* contains the data of each location.
Each column contains the information about the center of each possible location:
- ID
- latitude
- longitude

## Energy stations data
The file *substations_points.ssv* contains the data of each energy-supply station.
Each column contains the information about:
- latitude
- longitude
- maximun number of charging slots to which the energy station is capable of supplying energy, i.e., the number of electric cars charging simultaneously. 

