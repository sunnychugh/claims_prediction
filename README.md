# Claims_Prediction

This repository contains code of claims prediction for Hastings Direct take-home test.

### Directory Structure

```sh
.
├── regression.py                             # Class/Functions to do subtasks
├── main.py                                   # Main code file
├── README.md
├── requirements.txt                          # Packages required
├── config.py                                 # Store parameters value
└── data                                      # Various data files
    ├── Data_Scientist_Interview_Task.xlsx

```

- **config.py** file contains the various default variables value. Change the values as required.

### Requirements

- Python 3.7 or greater (preferable)

### Installation

#### 1. Conda Environment

Download and install Miniconda.

```sh
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ bash Miniconda3-latest-Linux-x86_64.sh
```

Create virtual environment, activate, and install packages.

```sh
$ conda create -n venv python=3.7
$ conda activate venv
$ pip install -r requirements.txt
```

#### 2. Docker Environment

You need to have Docker installed on your workstation. Installation process depends on the type of operating system (Windows, Mac, or Linux). Check online about how to install it.

### Run the code (Usage)

#### 1. Locally (on Workstation)

- Open a terminal and type/run:
  ```sh
  $ python main.py [-v] [-r <regressor name>]
  ```
  - Add `-r` or `--regressor` flag mentioning the name of the machine learning regressor used.
  - Add `-v` or `--verbose` flag to print the details (Pandas dataframe of the data, mean squared error, score, etc) on the terminal.
