#!/bin/bash

#----------------------------------------------
# Note that this is the standard way of doing 
# things in Python 3.6. Earlier versions used
# virtulalenv. It is best to convert to 3.6
# before you do anything else. 
# Note that the default Python version in 
# the AWS Ubuntu is 3.5 at the moment. You
# will need to upgrade the the new version 
# if you wish to use this environment in 
# AWS
#----------------------------------------------
python3 -m venv env

# this is for bash. Activate
# it differently for different shells
#--------------------------------------
source env/bin/activate 

pip3 install --upgrade pip

if [ -e requirements.txt ]; then

    pip3 install -r requirements.txt

else

    # basic utilities
    pip3 install --upgrade pytest
    pip3 install --upgrade pytest-cov
    pip3 install --upgrade sphinx
    pip3 install --upgrade sphinx_rtd_theme

    # Utilities
    pip3 install --upgrade ipython
    pip3 install --upgrade tqdm

    # scientific libraries
    pip3 install --upgrade numpy
    pip3 install --upgrade scipy

    # ML libraries
    pip3 install --upgrade sklearn
    pip3 install --upgrade tensorflow

    # Charting libraries
    pip3 install --upgrade matplotlib

    pip3 freeze > requirements.txt

fi

deactivate