import time
import json
import numpy
import pandas
import pathlib
import tensorflow
import scipy.optimize

import dataset_loader


class OptimizedEnsembledModel:


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

        self.name = "OEModel"
        self.title = title
        self.labels = labels
        self.image_size = image_size

        self.num_models = len(self.models)
        self.weights = numpy.ones(self.num_models) / self.num_models


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
        objective_function = lambda index: numpy.dot(losses, hyperplane[int(index[0])]) / self.num_models
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

        yp_ensembled = numpy.sum(yp_weighted, axis=0)

        return yp_ensembled


    def fit_report(self, directory: pathlib.Path, x_train: dataset_loader.DatasetLoader, x_val: dataset_loader.DatasetLoader) -> None:

        info = pandas.DataFrame(columns=["Identifier", "Backbone", "Fit Time", "Train Samples", "Validation Samples"])

        length_train = x_train.length()
        length_val = x_val.length()

        time_fit_beg = time.perf_counter()
        self.fit(x_train, x_val)
        time_fit_end = time.perf_counter()
        time_fit_dataset = time_fit_end - time_fit_beg

        info.loc[len(info)] = ["0", self.name, time_fit_dataset, length_train, length_val]
        info.to_csv(str(directory / "info.csv"), index=False)


    def save(self, directory: pathlib.Path) -> None:

        names = []
        for model in self.models:
            name = f"{model.name}.keras"
            path = str(directory / name)
            model.save(path)
            names.append(name)

        data = {
            "models": names,
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
    def load(cls, directory: pathlib.Path) -> "OptimizedEnsembledModel":

        with open(str(directory / "data.json"), mode="r") as file:
            data = json.load(file)

        models = []
        for name in data["models"]:
            path = str(directory / name)
            model = tensorflow.keras.models.load_model(path)
            models.append(model)

        oem = cls(data["title"], data["labels"], data["image_size"])
        oem.models = models
        oem.name = data["name"]
        oem.num_models = data["num_models"]
        oem.weights = numpy.array(data["weights"])

        return oem