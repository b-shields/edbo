# edbo

## Testing

This directory contains a number of tests of the basic functionality of *edbo*. Some of the functions test *edbo*'s plotting functions so you will need to close the plot windows as they pop up for tests to proceed.

### Running tests

Individual tests can be run using the pytest framework.

```bash
pytest <test>.py

# Note: Make sure pytest is installed (basic_tests.sh script installs pytest and runs a set of tests).
```
### Run a test of the basic functionality

```bash
cd ./edbo/tests
sh basic_tests.sh

# Note: If OS is Linux based you may need to chmod + tests.sh
```