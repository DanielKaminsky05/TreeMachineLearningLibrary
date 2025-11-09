[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/8Cy36LS2)
[![Work in MakeCode](https://classroom.github.com/assets/work-in-make-code-8824cc13a1a3f34ffcd245c82f0ae96fdae6b7d554b6539aec3a03a70825519c.svg)](https://classroom.github.com/online_ide?assignment_repo_id=21105021&assignment_repo_type=AssignmentRepo)
# Assignment 2 – Low-Fidelity Prototype

## What to Submit

- Upload your report in the `/docs` folder

## Folder Structure
- `/docs/report.pdf`

## Submission Notes
Push all files before the deadline. This repo is private and only visible to you and instructors.
# Synced update - Added ProjectTemplatee


## Datasets Used 

We used the Iris dataset and the Titanic Dataset from Kaggle to test our machine learning library. Python code was used for preprocessing, and can be found in 
`data-preprocessing/data-files`. The `data-files` directory includes `regression` and `classification` datasets, which contains already preprocessed and split datasets for regressive and classification tasks respectively.

The datasets were preprocessed using Numpy, Pandas and Scikit-learn in Python notebooks. The raw data, processed datasets and the code used to obtain the processed data are in `data-preprocessing`. Please refer to the [Readme](data-preprocessing/README.md) for more details.

# Classes API definition

## Dataset.cpp

The dataset.cpp loads a CSV in contiguous memory as a single homogenous 1D vector array of floats.

`read_csv`:
`get_path`:

### Usage: 

in `main.cpp` in the project root, load a CSV file in `data-preprocessing` into a Dataset object using `../data-preprocessing/` as the file path, since all the 
code is built inside `build/`.
