"""
Maintainer: Gabriel Dias (g172441@dac.unicamp.br)
            Mateus Oliveira (m203656@dac.unicamp.br)
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
from utils import read_yaml
from plot_metrics import PlotMetrics
from constants import *
from torch.utils.data import DataLoader
from metrics import calculate_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config_file", type=str, help="",
    )
    parser.add_argument(
        "weight", type=str, help="WEIGHTs neural network"
    )

    args = parser.parse_args()
    configs = read_yaml(args.config_file)
    load_dict = torch.load(args.weight)

    save_dir_path = "evaluate_results"
    model_configs = configs["model"]

    if type(model_configs) == dict:
        model = FACTORY_DICT["model"][list(model_configs.keys())[0]](
            **model_configs[list(model_configs.keys())[0]]
        )
    else:
        model = FACTORY_DICT["model"][model_configs]()

    model.load_state_dict(load_dict["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    test_dataset_configs = configs["test_dataset"]
    test_dataset = FACTORY_DICT["dataset"][list(test_dataset_configs)[0]](
        **test_dataset_configs[list(test_dataset_configs.keys())[0]])

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for i, dataset in enumerate(tqdm(test_loader)):
        input, target, ppm, filename = dataset
        filename = filename[0].split(".")[0]

        input = input.to(DEVICE)

        prediction = model(input)

        prediction = prediction.cpu().detach().numpy()
        target = target.numpy()
        ppm = ppm.numpy()

        result = calculate_metrics(prediction, target, ppm)

        if i == 0:
            result["filename"] = [filename]
            row_df = result
        else:
            mse = result['mse'][0]
            snr = result['snr'][0]
            linewidth = result['linewidth'][0]
            shape_score = result['shape_score'][0]

            row_df["mse"].append(mse)
            row_df["snr"].append(snr)
            row_df["linewidth"].append(linewidth)
            row_df["shape_score"].append(shape_score)
            row_df["filename"].append(filename)

        target = target[0, :]
        prediction = prediction[0, :]
        ppm = ppm[0, :]

        save_dir_path = "evaluate_results"

        os.makedirs(save_dir_path, exist_ok=True)
        os.makedirs(os.path.join(save_dir_path, "spectra_comparison"), exist_ok=True)
        os.makedirs(os.path.join(save_dir_path, "shape_score_comparison"), exist_ok=True)

        filename = filename.split(".")[0]
        PlotMetrics.spectra_comparison(prediction, target, ppm,
                                       f"{save_dir_path}/spectra_comparison/{filename}.png")
        PlotMetrics.shape_score_comparison(prediction, target, ppm,
                                           f"{save_dir_path}/shape_score_comparison/{filename}.png")

    df = pd.DataFrame(row_df)

    df.to_csv(os.path.join(save_dir_path, "result_metrics.csv"), index=False)

    print(f"Mean MSE: {df['mse'].mean()}")
    print(f"Mean SNR: {df['snr'].mean()}")
    print(f"Mean FHWM: {df['linewidth'].mean()}")
    print(f"Mean Shape Score: {df['shape_score'].mean()}")
