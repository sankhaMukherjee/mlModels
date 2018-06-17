# mlModels

This repo is going to contain a set of libraries usefull for quick deployment for different machine learning tasks. 

## 1. Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## 2. Prerequisites

You will need to have a valid Python installation on your system. This has been tested with Python 3.6. It does not assume a particulay version of python, however, it makes no assertions of proper working, either on this version of Python, or on another. The rest of the requirements is present in the file called 'requirements.txt'. 

Note: requirements.txt installs `tensorflow` by default. You may want to install `tensorflow-gpu` insteaad. Lateky, there seems to be a problem with the pip3 installation of `tensorflow-gpu` and you may need to compile from sources. If that is the case, then you will need to change the installation instructions accordingly.

## 2.1. Installing

The folloiwing installations are for *nix-like systems. These have been tried on macOS Sierra (Version 10.12.6) before.* 

1. Clone the program to your computer. 
2. type `make firstRun`. This should do the following
    2.1. generate a virtual environment in folder `env`
    2.2. install a number of packages
    2.3. generate a new `requirements.txt` file
    2.4. generate an initial git repository
3. change to the `src` folder
4. run the command `make run`. This should run the small test program
5. Generate your documentation folder by running `make doc`. 
6. Check whether all the tests pass with the command `make test`. This uses py.test for running tests. 

This code is organized in the folowing manner:

 - bin             : folder containing shell scripts
 - config          : folder containing JSON config files
   ├── config.json   : main configuration options.
   └── modules.json  : config files that allows different modeules to be executed.
 - data            : folder containing data used for training/testing
 - docs            : folder containing automatic documentation
 - env             : virtal environment
 - notebooks       : folder containing exploratory notebooks
 - src             : source files (main folders are listed below)
   ├── Makefile    : make file. Main uses are `make run`, `make doc`, `make test`
   ├── lib         : folder containing all the required libraries. Check the `Readme.md` for each library
   │   └── simpleLib.py
   ├── mlModels.py : main file that will be executed
   └── modules     : modules are independent pieces of code that will be automatically executed.
       └── module1 : this is an example module for reference. 
 - tests     : code for unit testing

## 3. Deployment

This code is not meant to be deployed. This is useful when you want to have your own implementation, and would like to add NN functionality quickly to your models. 

## 4. Built With

 - Python 3.6

## 5. Contributing

Please send in a pull request.

## 6. Authors

Sankha Mukherjee - Initial work (2018)

## 7. License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

## 8. Acknowledgments/References

 - this will be added later
 