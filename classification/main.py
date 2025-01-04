import click
import numpy
import gradio
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
    directory_results = pathlib.Path(f"{directory_template} - Model")


    if not directory_results.exists():

        directory_results.mkdir()

        dl_train, dl_test, dl_val, title, labels = dataset_loader.DatasetLoader.from_directory(directory_dataset, IMAGE_SIZE, BATCH_SIZE)

        model = optimized_ensembled_model.OptimizedEnsembledModel(title, labels, IMAGE_SIZE)
        model.fit_predict_evaluation(directory_results, dl_train, dl_test, dl_val)
        model.save(directory_results)

    else:

        model = optimized_ensembled_model.OptimizedEnsembledModel.load(directory_results)


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
