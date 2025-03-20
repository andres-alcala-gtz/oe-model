import click
import pathlib

import environment
import dataset_loader
import optimized_ensembled_model


if __name__ == "__main__":


    IMAGE_SIZE = environment.IMAGE_SIZE
    BATCH_SIZE = environment.BATCH_SIZE
    FIGURE_SIZE = environment.FIGURE_SIZE


    directory_template = click.prompt("Dataset Directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))

    directory_dataset = pathlib.Path(directory_template)
    directory_results = pathlib.Path(f"{directory_template} - Experimentation")


    if not directory_results.exists():

        directory_results.mkdir()


    index = len(list(directory_results.glob("*")))


    while True:

        index += 1
        experiment = f"Experiment {index}"

        print(f"\n{experiment.upper()}")

        directory_experiment = directory_results / experiment
        directory_experiment.mkdir()

        dl_train, dl_test, dl_val, title, labels = dataset_loader.DatasetLoader.from_directory(directory_dataset, IMAGE_SIZE, BATCH_SIZE)

        model = optimized_ensembled_model.OptimizedEnsembledModel(title, labels, IMAGE_SIZE)
        model.fit_predict_evaluation(directory_experiment, dl_train, dl_test, dl_val, FIGURE_SIZE)
        model.save(directory_experiment)
