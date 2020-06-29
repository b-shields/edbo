# edbo

Experimental Design via Bayesian Optimization

## Testing

This directory contains a number of tests of the basic functionality of *edbo*. Some of the functions test *edbo*'s plotting functions so you will need to close the plot windows as they pop up for tests to proceed.

### Running tests

Individual tests can be run using the pytest framework.

```bash
pytest <test>.py

# Note: Make sure pytest is installed (tests.sh installs pytest and runs all tests).
```

### Run all tests

Running all of the tests will take ~ 1h.

```bash
cd ./edbo/tests
sh tests.sh

# Note: If OS is Linux based you may need to chmod + tests.sh
```