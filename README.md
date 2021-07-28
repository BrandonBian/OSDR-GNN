# My modification of OSDR-GNN for customized prediction on materials

On line 18, change the global variable to indicate whether to train on materials or on teir functions (original implementation).

```
material = True
```

## (2021/07/27 Update): Added new dataset (v6) with new attribute "manufac_type"
1. Debugged the dataset so that the columns match the previous version (v3).
2. Modified the data processing code to take into account the new attribute (feature).


# Original Information
## ADS-OSU-Assembly-Graph
> Ongoing collaboration between Autodesk and Oregon State University concerning the creation of assembly-flow based knowledge graphs to infer function

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Based in Python

## Code Dependencies 

- networkX
- CSV
- numpy


## Meta

Vincenzo Ferrero -  Ferrerov@oregonstate.edu

Daniele Grandi - daniele.grandi@autodesk.com

Kaveh Hassani - kaveh.hassani@autodesk.com

Distributed under the MIT License. See ``LICENSE`` for more information.


## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch 
3. Commit your changes 
4. Push to the branch 
5. Create a new Pull Request
