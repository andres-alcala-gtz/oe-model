import time
import uuid
import click
import pandas
import pathlib
import tensorflow
import collections

import environment
import dataset_loader


if __name__ == "__main__":


    IMAGE_SIZE = environment.IMAGE_SIZE
    BATCH_SIZE = environment.BATCH_SIZE


    directory_dataset = click.prompt("Dataset Directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))

    directory_dataset = pathlib.Path(directory_dataset)
    directory_results = pathlib.Path(f"{directory_dataset} - Control")


    if not directory_results.exists():

        directory_results.mkdir()

        info = pandas.DataFrame(columns=["Identifier", "Backbone", "Fit Time", "Train Samples", "Validation Samples"])

    else:

        info = pandas.read_csv(str(directory_results / "info.csv"))


    _, _, _, labels = dataset_loader.DatasetLoader.from_directory(directory_dataset, IMAGE_SIZE, BATCH_SIZE)


    constructors = {
        constructor(IMAGE_SIZE, len(labels)).name: constructor
        for constructor in environment.MODELS
    }


    while True:

        index = len(info)

        counter = collections.Counter({name: 0 for name in constructors.keys()})
        counter.update(info["Backbone"])

        print(f"\nMODEL {index + 1}")

        dl_train, dl_val, _, _ = dataset_loader.DatasetLoader.from_directory(directory_dataset, IMAGE_SIZE, BATCH_SIZE)

        length_train = dl_train.length()
        length_val = dl_val.length()

        identifier = str(uuid.uuid4())
        name = min(counter, key=counter.get)

        model = constructors[name](IMAGE_SIZE, len(labels))

        print(f"{model.name.upper()}: FITTING")
        time_fit_beg = time.perf_counter()
        model.fit(x=dl_train, validation_data=dl_val, epochs=100, verbose=1, callbacks=tensorflow.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True))
        time_fit_end = time.perf_counter()
        time_fit_dataset = time_fit_end - time_fit_beg

        model.save(str(directory_results / f"{identifier}.keras"))

        info.loc[index] = [identifier, name, time_fit_dataset, length_train, length_val]
        info.to_csv(str(directory_results / "info.csv"), index=False)
