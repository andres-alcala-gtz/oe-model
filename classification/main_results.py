import click
import pandas
import pathlib

import benchmark
import environment


if __name__ == "__main__":


    FIGURE_SIZE = environment.FIGURE_SIZE


    directory_template = click.prompt("Dataset Directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))

    directory_results = pathlib.Path(f"{directory_template} - Results")
    directory_model = pathlib.Path(f"{directory_template} - Model")
    directory_control = pathlib.Path(f"{directory_template} - Control")

    directory_results.mkdir()


    info_model = pandas.read_excel(str(directory_model / "info.xlsx"))
    info_control = pandas.read_excel(str(directory_control / "info.xlsx"))

    info = pandas.concat([info_model, info_control], ignore_index=True)
    info.to_excel(str(directory_results / "info.xlsx"), index=False)


    benchmark.comparison(directory_results, info, FIGURE_SIZE)
