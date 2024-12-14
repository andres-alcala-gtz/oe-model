import time
import uuid
import click
import pandas
import pathlib
import tensorflow
import collections

import dataset_loader


if __name__ == "__main__":


    IMAGE_SIZE = 512
    BATCH_SIZE = 8


    dataset_directory = click.prompt("Dataset Directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))

    dataset_directory = pathlib.Path(dataset_directory)
    results_directory = pathlib.Path(f"{dataset_directory} - Control")


    if not results_directory.exists():

        results_directory.mkdir()

        info = pandas.DataFrame(columns=["Identifier", "Backbone", "Fit Time", "Train Samples", "Validation Samples"])

    else:

        info = pandas.read_csv(str(results_directory / "info.csv"))


    _, _, _, labels = dataset_loader.DatasetLoader.from_directory(dataset_directory, IMAGE_SIZE, BATCH_SIZE)


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


    while True:

        index = len(info)

        counter = collections.Counter({name: 0 for name in backbones.keys()})
        counter.update(info["Backbone"])

        print(f"\nMODEL {index + 1}")

        dl_train, dl_val, _, _ = dataset_loader.DatasetLoader.from_directory(dataset_directory, IMAGE_SIZE, BATCH_SIZE)

        length_train = dl_train.length()
        length_val = dl_val.length()

        identifier = str(uuid.uuid4())
        name = min(counter, key=counter.get)
        backbone = backbones[name]

        model = wrapper(name, backbone)

        print(f"{model.name.upper()}: FITTING")
        time_fit_beg = time.perf_counter()
        model.fit(x=dl_train, validation_data=dl_val, epochs=100, verbose=1, callbacks=tensorflow.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True))
        time_fit_end = time.perf_counter()
        time_fit_dataset = time_fit_end - time_fit_beg

        model.save(str(results_directory / f"{identifier}.keras"))

        info.loc[index] = [identifier, name, time_fit_dataset, length_train, length_val]
        info.to_csv(str(results_directory / "info.csv"), index=False)
