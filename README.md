# myqlm-contrib

## Installation
It is good practice to install packages inside a virtualenv:

Create a fresh virtual env:

`$ virtualenv qlm_env`

Activate the environment:

`$ source ./qlm_env/bin/activate`

Finally install the package:

`$ python3 setup.py install`

## Running tests
You might want to run Sabre's test to check that everything is correctly installed. This command will install pytest and run the tests:

`$ python3 setup.py test`

## Repository content

### Sabre algorithm

A simple implementation of [Sabre algorithm](https://dl.acm.org/doi/10.1145/3297858.3304023) to deal with qubits mapping problem.
*Implementation by Quentin Delamea*.
