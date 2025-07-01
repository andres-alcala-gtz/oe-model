import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow
import segmentation_models


EVALUATORS = [
    segmentation_models.metrics.IOUScore,
    segmentation_models.metrics.FScore,
    segmentation_models.losses.JaccardLoss,
    segmentation_models.losses.DiceLoss,
    segmentation_models.losses.CategoricalFocalLoss,
]


LOSS = segmentation_models.losses.JaccardLoss
METRIC = segmentation_models.metrics.IOUScore


def wrapper(name: str, backbone: tensorflow.keras.Model) -> tensorflow.keras.Model:
    model = tensorflow.keras.Sequential(name=name, layers=[
        tensorflow.keras.layers.InputLayer(input_shape=(None, None, 3)),
        tensorflow.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0),
        backbone,
    ])
    model.compile(
        optimizer="adam",
        loss=LOSS(),
        metrics=METRIC(),
    )
    return model


def InceptionV3(num_labels: int) -> tensorflow.keras.Model:
    name = "InceptionV3"
    backbone = segmentation_models.Unet(backbone_name="inceptionv3", input_shape=(None, None, 3), classes=num_labels, activation="softmax", encoder_weights="imagenet", encoder_freeze=True)
    model = wrapper(name, backbone)
    return model

def MobileNetV2(num_labels: int) -> tensorflow.keras.Model:
    name = "MobileNetV2"
    backbone = segmentation_models.Unet(backbone_name="mobilenetv2", input_shape=(None, None, 3), classes=num_labels, activation="softmax", encoder_weights="imagenet", encoder_freeze=True)
    model = wrapper(name, backbone)
    return model

def ResNet34V2(num_labels: int) -> tensorflow.keras.Model:
    name = "ResNet34V2"
    backbone = segmentation_models.Unet(backbone_name="resnet34", input_shape=(None, None, 3), classes=num_labels, activation="softmax", encoder_weights="imagenet", encoder_freeze=True)
    model = wrapper(name, backbone)
    return model


MODELS = [
    InceptionV3,
    MobileNetV2,
    ResNet34V2,
]
