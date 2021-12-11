# MPConstellation

MPConstellation is a group project for ME 231A at UC Berkeley. The goal of the project is to demonstrate model-predictive control (MPC) for a constellation of satellites.

## Setup

Code has been testd on Python 3.6 and 3.7.

You will need to install pyomo and the ipopt solver.

We used a `conda` environment:

`conda config --add channels conda-forge`
`conda config --set channel_priority strict`
`conda install -c conda-forge pyomo`
`conda install -c conda-forge ipopt`

If any of these commands produce errors, try updating `conda`:

`conda upgrade -n base conda`

You may need to remove the `conda-forge` channel to upgrade, as it can prevent the environment from being solved:

`conda config --remove channels conda-forge`

## Tests

To run a specific test method, use for example:

`python3 -m unittest test_simulator.TestSimulator.test_get_trajectory_ODE`


For a whole test suite:

`python3 test_satellite.py`


For all test suites (current working directory must be the repo directory):

`python3 -m unittest`

## Output and Visualization

Calling `Simulator.save_to_csv()` will save one trajectory CSV for each satellite in the most recent simulation. The filename includes a timestamp and the Satellite object's unqiue ID.

To visualize simulation CSV(s), either use the Python plotting functions or plot in MATLAB:

`visualizer('all')`

will plot all CSVs in directory, while

`visualizer('trajectory_xxx_yyy.csv')`

will plot a specific CSV.

