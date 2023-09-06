# Sota model

This repository provides a simple docker to forecast the Fashion dataset introduce in this [paper](https://arxiv.org/abs/2202.03224) with sota approaches available in the [neuralforecast package](https://nixtla.github.io/neuralforecast/). The repository is organized as follow:

 - [run/](run/): Directory containing a script to forecast a dataset in the directory [data/](data/) with the n-hits method.
 - [data/](data/): Directory gathering the dataset that will be forecasted.
 - [docker/](docker/): directory gathering the code to build a simple docker with the n-hits model. 

## How to predict with neuralforecast

First, You can copy a dataset that you want to forecast in the directory [data/](data/). For the format, refer to the example here [dataset_example.csv](data/dataset_example.csv)

Then, you should build, run and enter into the docker. In the main folder, run
```bash
make build run enter
```
Finally, run the following script to fit and predict the last year of the dataset that you put in the directory "data/". N-hits predictions will be save in the directory "/result/model_dir_tag"
:
- [run_model.py](run/run_model.py)
run
```bash
python run/run_model.py --help # display the default parameters and their description
python run/run_model.py --model_dir_tag test_dataset_example --dataset_name dataset_example.csv # train and predict the last year of the dataset name dataset_example.csv with the several sota models and save the result in the directory "/result/model_dir_tag".
```
It is possible that default parameters of the sota models have to be changed according to the proposed dataset. To parameters are assessed, the learning rate of the optimizer and the batch size used during the training process. To do so, run the following commande. Warning: many training will be done with this script. On large dataset, try to run this script on a sub-sample of time series.
```bash
python run/run_model_gridsearch.py --help # display the default parameters and their description
python run/run_model_gridsearch.py --model_dir_tag test_dataset_example --dataset_name dataset_example.csv # run a gridsearch for the learning rate and batch size paramaters on the proposed dataset.
```
