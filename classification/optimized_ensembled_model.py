import time
import json
import numpy
import pandas
import pathlib
import tensorflow
import scipy.optimize

import benchmark
import architecture
import dataset_loader


class OptimizedEnsembledModel:


    def __init__(self, title: str, labels: list[str], image_size: int) -> None:

        self.models = [
            constructor(image_size, len(labels))
            for constructor in architecture.MODELS
        ]

        self.name = "OEModel"
        self.title = title
        self.labels = labels
        self.image_size = image_size

        self.num_models = len(self.models)
        self.weights = numpy.ones(self.num_models) / self.num_models


    def predict(self, x: dataset_loader.DatasetLoader | numpy.ndarray, verbose=0) -> numpy.ndarray:

        yp_simple = [
            model.predict(x=x, verbose=verbose)
            for model in self.models
        ]

        yp_weighted = [
            weight * yp
            for weight, yp in zip(self.weights, yp_simple)
        ]

        yp_ensembled = numpy.sum(yp_weighted, axis=0)

        return yp_ensembled


    def fit(self, x_train: dataset_loader.DatasetLoader, x_val: dataset_loader.DatasetLoader) -> None:

        print("\nFITTING")
        for model in self.models:
            print(f"{model.name.upper()}: TRAINING")
            model.fit(x=x_train, validation_data=x_val, epochs=100, verbose=1, callbacks=[tensorflow.keras.callbacks.ReduceLROnPlateau(patience=2), tensorflow.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])

        y_val = x_val.y()
        loss = architecture.LOSS()

        losses = numpy.array([
            float(loss(y_val, model.predict(x=x_val, verbose=0)))
            for model in self.models
        ])

        def objective_function(x: numpy.ndarray) -> float:
            return numpy.dot(losses, x)

        print("\nOPTIMIZING")
        matrix = numpy.ones((1, self.num_models))
        bounds = [(0.1, 0.9)] * (self.num_models)
        constraints = [scipy.optimize.LinearConstraint(A=matrix, lb=1.0, ub=1.0)]
        optimization = scipy.optimize.differential_evolution(func=objective_function, bounds=bounds, constraints=constraints, strategy="best1bin", disp=True)
        self.weights = optimization.x
        print(f"weights={self.weights}")


    def fit_predict_evaluation(self, directory: pathlib.Path, x_train: dataset_loader.DatasetLoader, x_test: dataset_loader.DatasetLoader, x_val: dataset_loader.DatasetLoader, figure_size: float) -> None:

        info = []

        self.fit(x_train, x_val)

        for model in [self] + self.models:
            time_predict_beg = time.perf_counter()
            y_test = model.predict(x_test, verbose=0)
            time_predict_end = time.perf_counter()
            time_predict_dataset = time_predict_end - time_predict_beg
            data = benchmark.evaluation(directory, model.name, time_predict_dataset, y_test, x_train, x_test, x_val, self.labels, figure_size)
            info.append(data)

        info = pandas.DataFrame(info)
        info.to_excel(str(directory / "info.xlsx"), index=False)


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
            model = tensorflow.keras.models.load_model(path, compile=False)
            models.append(model)

        oem = cls(data["title"], data["labels"], data["image_size"])
        oem.models = models
        oem.name = data["name"]
        oem.num_models = data["num_models"]
        oem.weights = numpy.array(data["weights"])

        return oem
