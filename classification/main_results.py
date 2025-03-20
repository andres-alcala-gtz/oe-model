import click
import pandas
import pathlib

import benchmark
import environment


if __name__ == "__main__":


    FIGURE_SIZE = environment.FIGURE_SIZE


    directory_template = click.prompt("Dataset Directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))

    directory_experimentation = pathlib.Path(f"{directory_template} - Experimentation")
    directory_results = pathlib.Path(f"{directory_template} - Results")

    directory_results.mkdir()


    info = [
        pandas.read_excel(str(location / "info.xlsx"))
        for location in directory_experimentation.glob("*")
    ]

    info = pandas.concat(info, ignore_index=True)
    info.to_excel(str(directory_results / "info.xlsx"), index=False)


    benchmark.comparison(directory_results, info, FIGURE_SIZE)
