import numpy
import pandas
import pathlib
import scipy.stats
import matplotlib.pyplot

import regexp
import architecture
import dataset_loader


def evaluation(backbone: str, time_predict: float, yp_test: numpy.ndarray, xt_train: dataset_loader.DatasetLoader, xt_test: dataset_loader.DatasetLoader, xt_val: dataset_loader.DatasetLoader) -> dict[str, str | float]:

    yt = xt_test.y()

    yp = yp_test

    length_train = xt_train.length()
    length_test = xt_test.length()
    length_val = xt_val.length()

    data = {}

    data["Backbone"] = backbone
    data["Train Samples"] = length_train
    data["Test Samples"] = length_test
    data["Validation Samples"] = length_val
    data["Predict Time"] = time_predict
    data["Predict Time Per Sample"] = time_predict / length_test

    for constructor in architecture.EVALUATORS:
        evaluator = constructor()
        title = regexp.space_pascal_case(constructor.__name__)
        score = float(evaluator(yt, yp))
        data[title] = score

    return data


def information_plot(directory: pathlib.Path, information: pandas.DataFrame, figure_size: float) -> None:

    figure, axes = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(figure_size, figure_size))

    title = information.columns[-1]

    figure.suptitle(title)

    markers = ["o", "s", "p", "*", "H", "X", "D"]

    for marker, (backbone, group) in zip(markers, information.groupby("Backbone")):
        x = range(1, len(group) + 1)
        y = group[title]
        axes.plot(x, y, marker=marker, linestyle="--", label=backbone)

    models_experimental = information[information["Backbone"] == "OEModel"][title]
    models_control = information[information["Backbone"] != "OEModel"][title]
    t_statistic, p_value = scipy.stats.ttest_ind(models_experimental, models_control)

    axes.set_title(f"T-Statistic: {t_statistic:.4f}, P-Value: {p_value:.4f}")
    axes.set_xlabel("Experiment Number")
    axes.set_ylabel("Score")
    axes.grid(True)
    axes.legend()

    figure.savefig(str(directory / f"{title}.png"))


def information_metadata(directory: pathlib.Path, information: pandas.DataFrame) -> None:

    title = information.columns[-1]

    indexes = {}
    indexes |= {"All": information.index}
    indexes |= {"Control": information[information["Backbone"] != "OEModel"].index}
    indexes |= {backbone: group.index for backbone, group in information.groupby("Backbone")}

    metadata = pandas.DataFrame()

    for backbone, location in indexes.items():
        info = information.iloc[location][[title]]
        description = info.agg(["min", "max", "mean", "median", "sem", "std", "var"])
        description.columns = [backbone]
        metadata = pandas.concat([metadata, description], axis=1)

    path = directory / "metadata.xlsx"

    if not path.exists():
        pandas.DataFrame([{"Description": "This file contains descriptive metrics for each evaluation"}]).to_excel(str(path), sheet_name="Introduction", index=False)

    with pandas.ExcelWriter(str(path), mode="a") as writer:
        metadata.to_excel(writer, sheet_name=title)


def comparison(directory: pathlib.Path, information: pandas.DataFrame, figure_size: float) -> None:

    for title in information.columns:
        if pandas.to_numeric(information[title], errors="coerce").notna().all():
            info = information[["Backbone", title]]
            information_plot(directory, info, figure_size)
            information_metadata(directory, info)
