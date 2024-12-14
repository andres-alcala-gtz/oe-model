import click
import numpy
import gradio
import pathlib

import dataset_loader
import optimized_ensembled_model


if __name__ == "__main__":


    IMAGE_SIZE = 512
    BATCH_SIZE = 8


    dataset_directory = click.prompt("Dataset Directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))

    dataset_directory = pathlib.Path(dataset_directory)
    results_directory = pathlib.Path(f"{dataset_directory} - Model")


    if not results_directory.exists():

        results_directory.mkdir()

        dl_train, dl_val, title, labels = dataset_loader.DatasetLoader.from_directory(dataset_directory, IMAGE_SIZE, BATCH_SIZE)

        model = optimized_ensembled_model.OptimizedEnsembledModel(title, labels, IMAGE_SIZE)
        model.fit_report(results_directory, dl_train, dl_val)
        model.save(results_directory)

    else:

        model = optimized_ensembled_model.OptimizedEnsembledModel.load(results_directory)


    title = model.title
    labels = model.labels

    def predict(image: numpy.ndarray[int]) -> dict[str, float]:
        if image is not None:
            x = numpy.expand_dims(image, axis=0)
            y = model.predict(x)[0]
            return {label: probability for label, probability in zip(labels, y)}
        else:
            return {"", 0.0}

    interface = gradio.Interface(
        title=title,
        fn=predict,
        inputs=gradio.Image(),
        outputs=gradio.Label(),
    )

    print("\nLAUNCHING")
    interface.launch(
        share=True,
    )
