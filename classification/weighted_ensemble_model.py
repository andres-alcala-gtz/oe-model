import time
import json
import numpy
import scipy
import pandas
import pathlib
import matplotlib
import tensorflow
import sklearn.metrics

import dataset_loader


class WeightedEnsembleModel:


    def __init__(self, title: str, labels: list[str], image_size: int) -> None:

        def wrapper(name: str, backbone: tensorflow.keras.Model) -> tensorflow.keras.Model:
            model = tensorflow.keras.Sequential(name=name, layers=[
                tensorflow.keras.layers.InputLayer(input_shape=(None, None, 3)),
                tensorflow.keras.layers.Resizing(height=image_size, width=image_size),
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
            "InceptionV3": tensorflow.keras.applications.InceptionV3(input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet", pooling="max"),
            "MobileNetV2": tensorflow.keras.applications.MobileNetV2(input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet", pooling="max"),
            "ResNet50V2": tensorflow.keras.applications.ResNet50V2(input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet", pooling="max"),
        }

        for backbone in backbones.values():
            backbone.trainable = False

        self.models = [
            wrapper(name, backbone)
            for name, backbone in backbones.items()
        ]

        self.name = "WeightedEnsembleModel"
        self.title = title
        self.labels = labels
        self.image_size = image_size

        self.num_models = len(self.models)
        self.weights = numpy.zeros(self.num_models)


    def fit(self, x_train: dataset_loader.DatasetLoader, x_val: dataset_loader.DatasetLoader) -> None:

        y_val = x_val.y()
        axes = [numpy.linspace(0.20, 0.80, 100)] * self.num_models
        coordinates = numpy.array(numpy.meshgrid(*axes)).reshape(self.num_models, -1).T
        hyperplane = coordinates[coordinates.sum(axis=1) == 1.00]

        print("\nFITTING")
        for model in self.models:
            print(f"{model.name.upper()}: TRAINING")
            model.fit(x=x_train, validation_data=x_val, epochs=100, verbose=1, callbacks=tensorflow.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True))

        losses = numpy.array([
            float(tensorflow.keras.losses.SparseCategoricalCrossentropy()(y_val, model.predict(x=x_val, verbose=0)))
            for model in self.models
        ])

        print("\nOPTIMIZING")
        objective_function = lambda index: numpy.dot(losses, hyperplane[int(index[0])])
        optimization = scipy.optimize.differential_evolution(func=objective_function, bounds=[(0, len(hyperplane) - 1)], strategy="rand1bin", disp=True)
        self.weights = hyperplane[int(optimization.x[0])]


    def predict(self, x: dataset_loader.DatasetLoader | numpy.ndarray[int], verbose=None) -> numpy.ndarray[float]:

        yp_raw = [
            model.predict(x=x, verbose=0)
            for model in self.models
        ]

        yp_weighted = [
            weight * yp
            for weight, yp in zip(self.weights, yp_raw)
        ]

        yp_ensemble = numpy.sum(yp_weighted, axis=0)

        return yp_ensemble


    def benchmark(self, x_test: dataset_loader.DatasetLoader, directory: pathlib.Path) -> None:

        UNIT_SIZE = 8.0
        FIGURE_SIZE = 2 * UNIT_SIZE, 1 * UNIT_SIZE

        length = x_test.length()
        y_test = x_test.y()
        data = {}

        figure, axes = matplotlib.pyplot.subplots(nrows=1, ncols=1, figsize=FIGURE_SIZE)

        figure.suptitle("Benchmark")
        for weight, model in zip([numpy.sum(self.weights)] + list(self.weights), [self] + self.models):
            time_beg = time.perf_counter()
            yp_test = model.predict(x=x_test, verbose=0)
            time_end = time.perf_counter()
            time_dataset = time_end - time_beg
            time_image = time_dataset / length
            loss = float(tensorflow.keras.losses.SparseCategoricalCrossentropy()(y_test, yp_test))
            metric = float(tensorflow.keras.metrics.SparseCategoricalAccuracy()(y_test, yp_test))
            data[model.name] = {"Weight": f"{weight:.4f}", "Predict Time (Dataset)": f"{time_dataset:.4f}", "Predict Time (Image)": f"{time_image:.4f}", "Loss": f"{loss:.4f}", "Metric": f"{metric:.4f}"}
        dataframe = pandas.DataFrame.from_dict(data).transpose()
        axes.table(cellText=dataframe.values, rowLabels=dataframe.index, colLabels=dataframe.columns, cellLoc="center", rowLoc="center", colLoc="center", loc="center")
        axes.axis("off")

        figure.tight_layout()
        figure.savefig(str(directory / "benchmark.png"))


    def confusion_matrix(self, x_test: dataset_loader.DatasetLoader, directory: pathlib.Path) -> None:

        UNIT_SIZE = 4.0
        ROWS, COLS = self.num_models + 1, 2
        FIGURE_SIZE = COLS * UNIT_SIZE, ROWS * UNIT_SIZE

        y_test = x_test.y()

        figure, axes = matplotlib.pyplot.subplots(nrows=ROWS, ncols=COLS, figsize=FIGURE_SIZE)

        figure.suptitle("Confusion Matrix")
        for index, model in enumerate([self] + self.models):
            yp_test = model.predict(x=x_test, verbose=0).argmax(axis=1)
            axes[index, 0].set_title(f"{model.name}: Count")
            axes[index, 1].set_title(f"{model.name}: Percentage")
            sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_test, yp_test, normalize=None, values_format=None, cmap="Blues", colorbar=False, ax=axes[index, 0])
            sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_test, yp_test, normalize="true", values_format=".0%", cmap="Blues", colorbar=False, ax=axes[index, 1])

        figure.tight_layout()
        figure.savefig(str(directory / "confusion_matrix.png"))


    def save(self, directory: pathlib.Path) -> None:

        paths = {}
        for model in self.models:
            path = str(directory / f"{model.name}.keras")
            model.save(path)
            paths[model.name] = path

        data = {
            "models": paths,
            "name": self.name,
            "title": self.title,
            "labels": self.labels,
            "image_size": self.image_size,
            "num_models": self.num_models,
            "weights": list(self.weights),
        }

        with open(str(directory / "data.json"), mode="w") as file:
            json.dump(data, file, indent=4)


    @classmethod
    def load(cls, directory: pathlib.Path) -> "WeightedEnsembleModel":

        with open(str(directory / "data.json"), mode="r") as file:
            data = json.load(file)

        models = []
        for path in data["models"].values():
            model = tensorflow.keras.models.load_model(path)
            models.append(model)

        wem = cls(data["title"], data["labels"], data["image_size"])
        wem.models = models
        wem.name = data["name"]
        wem.num_models = data["num_models"]
        wem.weights = numpy.array(data["weights"])

        return wem
