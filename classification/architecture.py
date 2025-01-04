import tensorflow


LOSS = tensorflow.keras.losses.CategoricalCrossentropy
METRIC = tensorflow.keras.metrics.CategoricalAccuracy


def wrapper(name: str, backbone: tensorflow.keras.Model, image_size: int, num_labels: int) -> tensorflow.keras.Model:
    model = tensorflow.keras.Sequential(name=name, layers=[
        tensorflow.keras.layers.InputLayer(input_shape=(None, None, 3)),
        tensorflow.keras.layers.Resizing(height=image_size, width=image_size),
        tensorflow.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0),
        backbone,
        tensorflow.keras.layers.Dense(units=512, activation="relu"),
        tensorflow.keras.layers.Dense(units=512, activation="relu"),
        tensorflow.keras.layers.Dense(units=512, activation="relu"),
        tensorflow.keras.layers.Dense(units=512, activation="relu"),
        tensorflow.keras.layers.Dense(units=num_labels, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss=LOSS(),
        metrics=METRIC(),
    )
    return model


def InceptionV3(image_size: int, num_labels: int) -> tensorflow.keras.Model:
    name = "InceptionV3"
    backbone = tensorflow.keras.applications.InceptionV3(input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet", pooling="max")
    backbone.trainable = False
    model = wrapper(name, backbone, image_size, num_labels)
    return model

def MobileNetV2(image_size: int, num_labels: int) -> tensorflow.keras.Model:
    name = "MobileNetV2"
    backbone = tensorflow.keras.applications.MobileNetV2(input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet", pooling="max")
    backbone.trainable = False
    model = wrapper(name, backbone, image_size, num_labels)
    return model

def ResNet50V2(image_size: int, num_labels: int) -> tensorflow.keras.Model:
    name = "ResNet50V2"
    backbone = tensorflow.keras.applications.ResNet50V2(input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet", pooling="max")
    backbone.trainable = False
    model = wrapper(name, backbone, image_size, num_labels)
    return model


MODELS = [
    InceptionV3,
    MobileNetV2,
    ResNet50V2,
]
