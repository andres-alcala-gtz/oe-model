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
    DATA_AUGMENTATION = environment.DATA_AUGMENTATION


    directory_template = click.prompt("Dataset Directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))

    directory_dataset = pathlib.Path(directory_template)
    directory_model = pathlib.Path(f"{directory_template} - Model")


    if not directory_model.exists():

        directory_model.mkdir()


    if not any(directory_model.glob("*")):

        dl_train, dl_test, dl_val, title, labels = dataset_loader.DatasetLoader.from_directory(directory_dataset, IMAGE_SIZE, BATCH_SIZE, DATA_AUGMENTATION)

        model = optimized_ensembled_model.OptimizedEnsembledModel(title, labels, IMAGE_SIZE)
        model.fit_predict_evaluation(directory_model, dl_train, dl_test, dl_val)
        model.save(directory_model)

    else:

        model = optimized_ensembled_model.OptimizedEnsembledModel.load(directory_model)


    title = model.title

    def predict(image: numpy.ndarray, probability: bool) -> numpy.ndarray:
        x = numpy.expand_dims(image, axis=0)
        y = model.predict(x) if not probability else model.predict_probability(x)
        return y[0, :, :, -1]

    interface = gradio.Interface(
        title=title,
        fn=predict,
        inputs=[gradio.Image(label="Image"), gradio.Checkbox(label="Probability")],
        outputs=[gradio.Image(label="Mask")],
    )

    print("\nLAUNCHING")
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
    )
