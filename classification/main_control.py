import time
import uuid
import click
import pandas
import pathlib
import tensorflow
import collections

import environment
import architecture
import dataset_loader


if __name__ == "__main__":


    IMAGE_SIZE = environment.IMAGE_SIZE
    BATCH_SIZE = environment.BATCH_SIZE
    FIGURE_SIZE = environment.FIGURE_SIZE


    directory_template = click.prompt("Dataset Directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))

    directory_dataset = pathlib.Path(directory_template)
    directory_results = pathlib.Path(f"{directory_template} - Control")


    if not directory_results.exists():

        directory_results.mkdir()

        info = pandas.DataFrame(columns=["Backbone"])

    else:

        info = pandas.read_excel(str(directory_results / "info.xlsx"))


    _, _, _, _, labels = dataset_loader.DatasetLoader.from_directory(directory_dataset, IMAGE_SIZE, BATCH_SIZE)


    constructors = {
        constructor.__name__: constructor
        for constructor in architecture.MODELS
    }


    while True:

        counter = collections.Counter({backbone: 0 for backbone in constructors.keys()})
        counter.update(info["Backbone"])

        print(f"\nMODEL {len(info) + 1}")

        dl_train, dl_test, dl_val, _, _ = dataset_loader.DatasetLoader.from_directory(directory_dataset, IMAGE_SIZE, BATCH_SIZE)

        backbone = min(counter, key=counter.get)
        identifier = str(uuid.uuid4())

        model = constructors[backbone](IMAGE_SIZE, len(labels))

        print(f"{model.name.upper()}: FITTING")
        time_fit_beg = time.perf_counter()
        model.fit(x=dl_train, validation_data=dl_val, epochs=100, verbose=1, callbacks=tensorflow.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True))
        time_fit_end = time.perf_counter()
        time_fit_dataset = time_fit_end - time_fit_beg

        print(f"{model.name.upper()}: PREDICTING")
        time_predict_beg = time.perf_counter()
        y_test = model.predict(x=dl_test, verbose=1)
        time_predict_end = time.perf_counter()
        time_predict_dataset = time_predict_end - time_predict_beg

        model.save(str(directory_results / f"{backbone}_{identifier}.keras"))

        data = {
            "Backbone": backbone,
            "Identifier": identifier,
            "Fit Time": time_fit_dataset,
            "Predict Time": time_predict_dataset,
        }
        data = pandas.DataFrame([data])

        info = pandas.concat([info, data], ignore_index=True)
        info.sort_values("Backbone", ignore_index=True, inplace=True)
        info.to_excel(str(directory_results / "info.xlsx"), index=False)
