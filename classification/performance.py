import time
import json
import click
import numpy
import pandas
import pathlib
import matplotlib
import tensorflow

import dataset_loader


def save_score(data: dict[str, dict[str, list[float]]], directory: pathlib.Path) -> None:

    UNIT_SIZE = 8.0
    FIGURE_SIZE = 2 * UNIT_SIZE, 2 * UNIT_SIZE

    figure, axes = matplotlib.pyplot.subplots(nrows=2, ncols=2, figsize=FIGURE_SIZE)

    figure.suptitle("Score")
    for name, score in data.items():
        for index, (key, values) in enumerate(score.items()):
            values_avg = numpy.mean(values)
            values_min = numpy.min(values)
            values_max = numpy.max(values)
            axes[index // 2, index % 2].set_title(key)
            axes[index // 2, index % 2].plot(values, label=f"{name} (avg={values_avg:.2f}, min={values_min:.2f}, max={values_max:.2f})")
            axes[index // 2, index % 2].grid(True)
            axes[index // 2, index % 2].legend()

    figure.tight_layout()
    figure.savefig(str(directory / "score.png"))

    with open(str(directory / "score.json"), mode="w") as file:
        json.dump(data, file, indent=4)


def save_benchmark(data: dict[str, dict[str, str]], directory: pathlib.Path) -> None:

    UNIT_SIZE = 8.0
    FIGURE_SIZE = 2 * UNIT_SIZE, 1 * UNIT_SIZE

    figure, axes = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=FIGURE_SIZE)

    figure.suptitle("Benchmark")
    dataframe = pandas.DataFrame.from_dict(data).transpose()
    axes.table(cellText=dataframe.values, rowLabels=dataframe.index, colLabels=dataframe.columns, cellLoc="center", rowLoc="center", colLoc="center", loc="center")
    axes.axis("off")

    figure.tight_layout()
    figure.savefig(str(directory / "benchmark.png"))

    with open(str(directory / "benchmark.json"), mode="w") as file:
        json.dump(data, file, indent=4)


def save_models(data: dict[str, tensorflow.keras.Model], directory: pathlib.Path) -> None:

    for model in data.values():
        model.save(str(directory / f"{model.name}.keras"))


if __name__ == "__main__":


    IMAGE_SIZE = 512
    BATCH_SIZE = 8


    dataset_directory = click.prompt("Dataset Directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))

    dataset_directory = pathlib.Path(dataset_directory)
    results_directory = pathlib.Path(f"{dataset_directory} - Performance")

    results_directory.mkdir()


    _, _, _, _, labels = dataset_loader.DatasetLoader.from_directory(dataset_directory, IMAGE_SIZE, BATCH_SIZE)


    def wrapper(name: str, backbone: tensorflow.keras.Model) -> tensorflow.keras.Model:
        model = tensorflow.keras.Sequential(name=name, layers=[
            tensorflow.keras.layers.InputLayer(input_shape=(None, None, 3)),
            tensorflow.keras.layers.Resizing(height=IMAGE_SIZE, width=IMAGE_SIZE),
            tensorflow.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0),
            backbone,
            tensorflow.keras.layers.Dense(units=512, activation="relu"),
            tensorflow.keras.layers.Dense(units=512, activation="relu"),
            tensorflow.keras.layers.Dense(units=512, activation="relu"),
            tensorflow.keras.layers.Dense(units=512, activation="relu"),
            tensorflow.keras.layers.Dense(units=len(labels), activation="softmax"),
        ])
        model.compile(
            optimizer="adam",
            loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(),
            metrics=tensorflow.keras.metrics.SparseCategoricalAccuracy(),
        )
        return model

    backbones = {
        "InceptionV3": tensorflow.keras.applications.InceptionV3(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights="imagenet", pooling="max"),
        "MobileNetV2": tensorflow.keras.applications.MobileNetV2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights="imagenet", pooling="max"),
        "ResNet50V2": tensorflow.keras.applications.ResNet50V2(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights="imagenet", pooling="max"),
    }

    for backbone in backbones.values():
        backbone.trainable = False


    data_score = {}
    data_benchmark = {}
    data_models = {}

    for name in backbones.keys():
        data_score[name] = {"Fit Time": [], "Predict Time": [], "Loss": [], "Metric": []}
        data_benchmark[name] = {}
        data_models[name] = None


    for index in range(5):

        print(f"\nROUND {index + 1}")

        dl_train, dl_test, dl_val, _, _ = dataset_loader.DatasetLoader.from_directory(dataset_directory, IMAGE_SIZE, BATCH_SIZE)

        length_train = dl_train.length()
        length_test = dl_test.length()

        y_train = dl_train.y()
        y_test = dl_test.y()

        models = [wrapper(name, backbone) for name, backbone in backbones.items()]

        for model in models:

            print(f"{model.name.upper()}: FITTING")
            time_fit_beg = time.perf_counter()
            model.fit(x=dl_train, validation_data=dl_val, epochs=100, verbose=1, callbacks=tensorflow.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True))
            time_fit_end = time.perf_counter()
            time_fit_dataset = time_fit_end - time_fit_beg
            time_fit_image = time_fit_dataset / length_train

            print(f"{model.name.upper()}: PREDICTING")
            time_predict_beg = time.perf_counter()
            yp_test = model.predict(x=dl_test, verbose=1)
            time_predict_end = time.perf_counter()
            time_predict_dataset = time_predict_end - time_predict_beg
            time_predict_image = time_predict_dataset / length_test

            loss = float(tensorflow.keras.losses.SparseCategoricalCrossentropy()(y_test, yp_test))
            metric = float(tensorflow.keras.metrics.SparseCategoricalAccuracy()(y_test, yp_test))

            if len(data_score[model.name]["Loss"]) == 0 or min(data_score[model.name]["Loss"]) > loss:

                data_benchmark[model.name] = {
                    "Fit Time (Dataset)": f"{time_fit_dataset:.4f}",
                    "Fit Time (Image)": f"{time_fit_image:.4f}",
                    "Predict Time (Dataset)": f"{time_predict_dataset:.4f}",
                    "Predict Time (Image)": f"{time_predict_image:.4f}",
                    "Loss": f"{loss:.4f}",
                    "Metric": f"{metric:.4f}",
                }

                data_models[model.name] = model

            data_score[model.name]["Fit Time"].append(time_fit_dataset)
            data_score[model.name]["Predict Time"].append(time_predict_dataset)
            data_score[model.name]["Loss"].append(loss)
            data_score[model.name]["Metric"].append(metric)


    save_score(data_score, results_directory)
    save_benchmark(data_benchmark, results_directory)
    save_models(data_models, results_directory)
