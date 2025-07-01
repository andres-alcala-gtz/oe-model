import click
import pandas
import pathlib

import regexp
import benchmark
import environment
import dataset_loader
import optimized_ensembled_model


if __name__ == "__main__":


    IMAGE_SIZE = environment.IMAGE_SIZE
    BATCH_SIZE = environment.BATCH_SIZE
    EXPERIMENT_RUNS = environment.EXPERIMENT_RUNS
    FIGURE_SIZE = environment.FIGURE_SIZE
    DATA_AUGMENTATION = environment.DATA_AUGMENTATION


    directory_template = click.prompt("Dataset Directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))

    directory_dataset = pathlib.Path(directory_template)
    directory_experimentation = pathlib.Path(f"{directory_template} - Experimentation")

    directory_results = directory_experimentation / "Results"


    if not directory_experimentation.exists():

        directory_experimentation.mkdir()


    indexes_finished = []

    for experiment in directory_experimentation.glob("*"):
        if any(experiment.glob("*")):
            index = regexp.fetch_number(experiment.stem)
            indexes_finished.append(index)
        else:
            experiment.rmdir()

    indexes_expected = set(range(1, EXPERIMENT_RUNS + 1))
    indexes_finished = set(indexes_finished)
    indexes = sorted(indexes_expected - indexes_finished)


    for index in indexes:

        experiment = f"Experiment {index}"

        print(f"\n{experiment.upper()}/{EXPERIMENT_RUNS}")

        directory_experiment = directory_experimentation / experiment
        directory_experiment.mkdir()

        dl_train, dl_test, dl_val, title, labels = dataset_loader.DatasetLoader.from_directory(directory_dataset, IMAGE_SIZE, BATCH_SIZE, DATA_AUGMENTATION)

        model = optimized_ensembled_model.OptimizedEnsembledModel(title, labels, IMAGE_SIZE)
        model.fit_predict_evaluation(directory_experiment, dl_train, dl_test, dl_val)
        model.save(directory_experiment)


    info = [
        pandas.read_excel(str(experiment / "info.xlsx"))
        for experiment in sorted(directory_experimentation.glob("*"), key=lambda path: regexp.fetch_number(path.stem))
    ]


    if not directory_results.exists():

        directory_results.mkdir()


    info = pandas.concat(info, ignore_index=True)
    info.to_excel(str(directory_results / "info.xlsx"), index=False)


    benchmark.comparison(directory_results, info, FIGURE_SIZE)
