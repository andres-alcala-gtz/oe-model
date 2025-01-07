import re
import numpy
import pandas
import pathlib
import sklearn.metrics
import matplotlib.pyplot

import architecture
import dataset_loader


def roc_curve(directory: pathlib.Path, backbone: str, identifier: str, y_true: numpy.ndarray[float], y_pred: numpy.ndarray[float], labels: list[str], figure_size: float) -> None:

    figure, axes = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(figure_size, figure_size))

    figure.suptitle(backbone)

    for index in range(len(labels)):
        sklearn.metrics.RocCurveDisplay.from_predictions(y_true[:, index], y_pred[:, index], name=labels[index], ax=axes)

    figure.savefig(str(directory / f"{backbone}_{identifier}_RocCurve.png"))


def confusion_matrix(directory: pathlib.Path, backbone: str, identifier: str, y_true: numpy.ndarray[float], y_pred: numpy.ndarray[float], labels: list[str], figure_size: float) -> None:

    figure, axes = matplotlib.pyplot.subplots(nrows=1, ncols=2, figsize=(2 * figure_size, figure_size))

    figure.suptitle(backbone)

    sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize=None, values_format=None, display_labels=labels, cmap="Blues", colorbar=False, ax=axes[0])
    sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize="true", values_format=".0%", display_labels=labels, cmap="Blues", colorbar=False, ax=axes[1])

    figure.savefig(str(directory / f"{backbone}_{identifier}_ConfusionMatrix.png"))


def evaluation(directory: pathlib.Path, backbone: str, identifier: str, time_fit: float, time_predict: float, yp_test: numpy.ndarray[float], xt_train: dataset_loader.DatasetLoader, xt_test: dataset_loader.DatasetLoader, xt_val: dataset_loader.DatasetLoader, labels: list[str], figure_size: float) -> dict[str, str | float]:

    yt_categorical = xt_test.y()
    yt_numerical = numpy.argmax(yt_categorical, axis=1)

    yp_categorical = yp_test
    yp_numerical = numpy.argmax(yp_categorical, axis=1)

    length_train = xt_train.length()
    length_test = xt_test.length()
    length_val = xt_val.length()

    data = {}

    data["Backbone"] = backbone
    data["Identifier"] = identifier
    data["Fit Time"] = time_fit
    data["Predict Time"] = time_predict
    data["Train Samples"] = length_train
    data["Test Samples"] = length_test
    data["Validation Samples"] = length_val

    data["Fit Time Per Sample"] = time_fit / (length_train + length_val)
    data["Predict Time Per Sample"] = time_predict / (length_test)

    for constructor in architecture.EVALUATORS_CATEGORICAL:
        evaluator = constructor()
        title = re.sub(r"([a-z])([A-Z])", r"\1 \2", constructor.__name__)
        score = float(evaluator(yt_categorical, yp_categorical))
        data[title] = score

    for constructor in architecture.EVALUATORS_NUMERICAL:
        evaluator = constructor()
        title = re.sub(r"([a-z])([A-Z])", r"\1 \2", constructor.__name__)
        score = float(evaluator(yt_numerical, yp_numerical))
        data[title] = score

    roc_curve(directory, backbone, identifier, yt_categorical, yp_categorical, labels, figure_size)
    confusion_matrix(directory, backbone, identifier, yt_numerical, yp_numerical, labels, figure_size)

    return data


def information_plot(directory: pathlib.Path, information: pandas.DataFrame, figure_size: float) -> None:

    figure, axes = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=(figure_size, figure_size))

    title = information.columns[-1]

    figure.suptitle(title)

    for backbone, group in information.groupby("Backbone"):

        x = group.index
        y = group[title]

        if group["Identifier"].isnull().any():
            axes.axhline(y[0], label=backbone, color="indigo")
        else:
            axes.scatter(x, y, label=backbone)

    axes.set_xticks([])
    axes.set_xlabel("Control Group")
    axes.set_ylabel("Score")
    axes.grid(axis="y")
    axes.legend()

    figure.savefig(str(directory / f"{title}.png"))


def information_metadata(directory: pathlib.Path, information: pandas.DataFrame) -> None:

    title = information.columns[-1]

    indexes = {}
    indexes |= {"All": information.index}
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

            info = information[["Backbone", "Identifier", title]]

            information_plot(directory, info, figure_size)
            information_metadata(directory, info)
