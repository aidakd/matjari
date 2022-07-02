### UPDATE FOR MATJARI!!!!!!!!
## From Notebook to package 🎁

It is time to move away from Jupyter Notebook, and start writing reusable code with python packages and modules.

In this challenge we will not implement new functionalities. We will reorganise the existing code into packages and modules.

### Package structure 🗺

We have created for you the following package structure:

```bash
.
└── TaxiFareModel
    ├── data.py       # functions to retrieve and clean data
    ├── encoders.py   # custom encoders and transformers for the pipeline
    ├── trainer.py    # main class that will build and train the pipeline
    └── utils.py      # utility functions
```

### Setup your package ⚙️

Let's create a packaged project from the code of the notebook. In order to achieve this, we provide you with a minimal template generator package: [`packgenlite`](https://github.com/krokrob/packgenlite).

- Install the `packgenlite` package from GitHub

``` bash
pip install git+ssh://git@github.com/krokrob/packgenlite
```

- Create a new packaged project called `TaxiFareModel` in your *projects directory*: `~/code/<user.github_nickname>`.

``` bash
cd ~/code/<user.github_nickname>
packgenlite TaxiFareModel
```

- Copy the code that we provide into your project

<details>
  <summary markdown='span'><strong>💡 How to copy the code from the challenge to the packaged project ? </strong></summary>


```bash
cp -r ~/code/<user.github_nickname>/<program.challenges_repo_name>/07-Data-Engineering/02-ML-Iteration/03-Notebook-to-package/*.py ~/code/<user.github_nickname>/TaxiFareModel/TaxiFareModel
```

</details>

- Make sure that your package has the following structure using the `tree` command:

``` bash
.
├── MANIFEST.in
├── Makefile
├── README.md
├── TaxiFareModel
│   ├── __init__.py
│   ├── data
│   ├── data.py
│   ├── encoders.py
│   ├── trainer.py
│   └── utils.py
├── notebooks
├── raw_data
├── requirements.txt
├── .gitignore
├── scripts
│   └── TaxiFareModel-run
├── setup.py
└── tests
    └── __init__.py
```
*You won't see the hidden files which start with a dot `.` - to see them, run the `tree` command with the `-a` flag which will show you all the files (`tree -a`). If you want to limit the amount of information output by the tree command, use the `-I` flag in order to ignore one or more patterns: `tree -a -I .git` to ignore the `.git` directory, `tree -a -I ".git|__pycache__"` to ignore several patterns: here the `.git` directory and the `__pycache__` directories*

👍 Your package is ready to be implemented!

### Download the dataset locally

- Download the `train.csv` and `test.csv` datasets from [Kaggle (`Data` tab)](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data)
- Move them under the `raw_data` folder
- Download 2 subsets of the full `train.csv` dataset:
  - [train_1k.csv](https://wagon-public-datasets.s3.amazonaws.com/taxi-fare-ny/train_1k.csv) with 1_000 rows
  - [train_10k.csv](https://wagon-public-datasets.s3.amazonaws.com/taxi-fare-ny/train_10k.csv) with 10_000 rows
- Move them to the `raw_data` folder
- Make sure your package has the following structure:

```bash
.
├── MANIFEST.in
├── Makefile
├── README.md
├── TaxiFareModel
│   ├── __init__.py
│   ├── data
│   ├── data.py
│   ├── encoders.py
│   ├── trainer.py
│   └── utils.py
├── notebooks
├── raw_data
│   ├── test.csv
│   ├── train.csv
│   ├── train_10k.csv
│   └── train_1k.csv
├── requirements.txt
├── .gitignore
├── scripts
│   └── TaxiFareModel-run
├── setup.py
└── tests
    └── __init__.py
```

Now that everything is set, let's inspect the content of the provided files...

### `data.py`

Inspect the provided `get_data` and `clean_data` functions.

### `utils.py`

Inspect the provided `haversine_vectorized` and `compute_rmse` functions.

You can store the `haversine_distance` function here if you use it.

### `encoders.py`

Let's store the custom encoders and transformers for distance and time features here.

This code file will store all the custom pipeline preprocessing blocks. Meaning the `DistanceTransformer` and `TimeFeaturesEncoder`.

### `trainer.py`

Implement the main class here.

The `Trainer` class is the main class. It should have:
- an `__init__` method called when the class is instanciated
- a `set_pipeline` method that builds the pipeline
- a `run` method that trains the pipeline
- an `evaluate` method evaluating the model

Make sure that you are confident with the following notions:
- attributes and methods of a class
- the `**kwargs` argument of a function and how to use it, (help [HERE](https://www.programiz.com/python-programming/args-and-kwargs) if unclear)

If you are not confident with any of these elements or the general structure of the code, ask for a TA.

### Test your packaged project

Once you have everything implemented, test that your packaged project works by running:

```bash
python -m TaxiFareModel.trainer
```

Or

```bash
python -i TaxiFareModel/trainer.py
```

### Hints for debugging 🐛

- Do not hesitate to breakdown your code into smaller function calls for debugging
- Use [if \_\_name__ == '\_\_main__'](https://www.geeksforgeeks.org/what-does-the-if-__name__-__main__-do/) at the end of each `.py` file in order to debug
- For instance to debug `data.py`, add at the end:

``` python
# TaxiFareModel/data.py

if __name__ == '__main__':
    df = get_data()
```

Then in an `ipython` session from your terminal, you can run:

```bash
In [1]: %run TaxiFareModel/data.py

In [2]: df.shape
```

### Install your packaged project

When your project is all set, you should install it locally so that its packages can be imported anywhere. From the directory of your project, run:

```bash
pip install -e .
```

Then you can open the `taxi_fare_model_package_testing.ipynb` and run the cells.

### Push your package on GitHub 🐙😸

1. Create a repository on GitHub
2. Add a remote between your package and the GitHub repository
3. Push your code on the GitHub repository

👍 Good job!
