import click
import numpy
import gradio
import pathlib

import dataset_loader
import weighted_ensemble_model


if __name__ == "__main__":


    IMAGE_SIZE = 512
    BATCH_SIZE = 8


    dataset_directory = click.prompt("Dataset Directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))

    dataset_directory = pathlib.Path(dataset_directory)
    results_directory = pathlib.Path(f"{dataset_directory} - Results")


    if not results_directory.exists():

        results_directory.mkdir()

        dl_train, dl_test, dl_val, title, labels = dataset_loader.DatasetLoader.from_directory(dataset_directory, IMAGE_SIZE, BATCH_SIZE)

        wem = weighted_ensemble_model.WeightedEnsembleModel(title, labels, IMAGE_SIZE)
        wem.fit(dl_train, dl_val)
        wem.benchmark(dl_test, results_directory)
        wem.confusion_matrix(dl_test, results_directory)
        wem.save(results_directory)

    else:

        wem = weighted_ensemble_model.WeightedEnsembleModel.load(results_directory)


    title = wem.title
    labels = wem.labels

    def predict(image: numpy.ndarray[int]) -> dict[str, float]:
        if image is not None:
            x = numpy.expand_dims(image, axis=0)
            y = wem.predict(x)[0]
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
