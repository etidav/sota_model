import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS, PatchTST
from utils.utils import write_json
import torch


def format_single_ts(ts, ts_index):
    formated_ts = pd.DataFrame(columns=["unique_id", "ds", "y"], index=range(len(ts)))
    formated_ts["unique_id"] = ts_index * 1.0
    formated_ts["ds"] = pd.to_datetime(ts.index)
    formated_ts["y"] = ts.values
    return formated_ts


def format_multiple_ts(data, horizon):
    train_data = data.iloc[:-horizon]
    test_data = data.iloc[-horizon:]
    formated_train_data = []
    formated_test_data = []
    for i, j in enumerate(data):
        formated_train_data.append(format_single_ts(train_data[j], i + 1))
        formated_test_data.append(format_single_ts(test_data[j], i + 1))
    formated_train_data = pd.concat(formated_train_data, axis=0).reset_index(drop=True)
    formated_test_data = pd.concat(formated_test_data, axis=0).reset_index(drop=True)

    return formated_train_data, formated_test_data


def evaluate(
    ground_truth, prediction, metrics,season
):
    """
    Evaluate the prediction of a HERMES model

    Arguments:

    - *ground_truth*: a pd.DataFrame with the historical data and the ground truth of the forecasted timeframe 
    - *prediction*: a pd.DataFrame with the prediction of the model
    - *metrics*: A list containing a list of metrics to compute

    Returns:

    - *model_eval*: a dict containing the metrics values
   """
    time_split = prediction.index[0]
    horizon = prediction.shape[0]
    histo_ground_truth = ground_truth.loc[:time_split].iloc[:-1]
    ground_truth = ground_truth.loc[time_split:].iloc[:horizon]

    if type(metrics) == str:
        metrics = [metrics]

    model_eval = {}
    for metric in metrics:
        if metric == "mase":
            denominator = np.mean(
                np.abs(
                    histo_ground_truth.values[season:] - histo_ground_truth.values[:-season]
                ),
                axis=0,
            )
            numerator = np.mean(np.abs(ground_truth.values - prediction.values), axis=0)
            mase = (numerator / denominator).mean()
            model_eval["mase"] = mase
        else:
            print(f"{metric} not implemented yet --> implemented metrics are {metrics}")

    return model_eval


def main():

    parser = argparse.ArgumentParser(description="train sota models on a dataset")
    parser.add_argument(
        "--model_dir_tag",
        type=str,
        help="Name of the directory where the model will be store",
        required=True,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="name of the dataset to use in the dir /model/data",
        required=True,
    )
    parser.add_argument(
        "--horizon", type=int, help="forecast horizon of the sota models", default=52,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help="max number of epochs that will be performed",
        default=100,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="learning rate of the optimizer",
        default=0.0005,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="number of different series in each batch",
        default=64,
    )
    parser.add_argument(
        "--seed", type=int, help="number of different series in each batch", default=42,
    )
    parser.add_argument(
        "--gpu", type=int, help="number of the gpu that will be used", default=0,
    )
    args = parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    model_dir_tag = args.model_dir_tag
    model_folder = os.path.join("/model/result/", model_dir_tag + "_" + str(seed))
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    dataset_path = os.path.join("/model/data", args.dataset_name)
    dataset = pd.read_csv(dataset_path, index_col=0)
    horizon = args.horizon
    max_steps = args.max_steps
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    normalized_dataset = (dataset - dataset.iloc[:52].mean()) / dataset.iloc[:52].std()
    y_train, y_test = format_multiple_ts(normalized_dataset, horizon)

    models = [
        NBEATS(
            input_size=2 * horizon,
            h=horizon,
            max_steps=max_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            random_seed=seed,
        ),
        NHITS(
            input_size=2 * horizon,
            h=horizon,
            max_steps=max_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            random_seed=seed,
        ),
        PatchTST(
            input_size=2 * horizon,
            h=horizon,
            max_steps=max_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            random_seed=seed,
        )
    ]
    nforecast = NeuralForecast(models=models, freq="W")
    nforecast.fit(df=y_train)
    nforecast.save(path=os.path.join(model_folder, "model_save"))
    y_pred_df = nforecast.predict().reset_index()
    for model_name in y_pred_df:
        if model_name in ['unique_id', 'ds']:
            continue
        model_prediction = y_pred_df[model_name].values.reshape(
            dataset.shape[1], horizon
        ) * dataset.iloc[:52].std().values.reshape((-1, 1)) + dataset.iloc[
            :52
        ].mean().values.reshape(
            (-1, 1)
        )
        model_prediction = pd.DataFrame(
            model_prediction.T,
            columns=dataset.columns,
            index=dataset.index[-horizon:],
        )
        model_prediction.to_csv(os.path.join(model_folder, f"{model_name.lower()}_prediction.csv"))

        model_eval = evaluate(
            ground_truth=dataset, prediction=model_prediction, metrics=["mase"], season=52
        )
        write_json(model_eval, os.path.join(model_folder, f"{model_name.lower()}_results.json"))

if __name__ == "__main__":

    main()
