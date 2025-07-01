import math
import numpy
import pathlib
import PIL.Image
import tensorflow
import sklearn.model_selection


class DatasetLoader(tensorflow.keras.utils.Sequence):


    def __init__(self, xl_set: list[pathlib.Path], yl_set: list[pathlib.Path], labels: list[int], image_size: int, batch_size: int, data_augmentation: bool, **kwargs) -> None:

        super().__init__(**kwargs)

        self.xl_set = xl_set
        self.yl_set = yl_set
        self.labels = labels
        self.image_size = image_size
        self.batch_size = batch_size
        self.data_augmentation = data_augmentation

        self.num_labels = len(self.labels)
        self.label_to_index = {label: index for index, label in enumerate(self.labels)}


    def __len__(self) -> int:

        return math.ceil(len(self.xl_set) / self.batch_size)


    def _augment(self, image: numpy.ndarray, mask: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:

        x, y = image, mask

        if numpy.random.rand() > 0.5:
            random = numpy.random.uniform(-0.1, 0.1)
            x = numpy.array(tensorflow.image.adjust_brightness(x, random))

        if numpy.random.rand() > 0.5:
            random = numpy.random.uniform(0.9, 1.1)
            x = numpy.array(tensorflow.image.adjust_contrast(x, random))

        if numpy.random.rand() > 0.5:
            random = int(numpy.random.uniform(90, 100))
            x = numpy.array(tensorflow.image.adjust_jpeg_quality(x, random))

        if numpy.random.rand() > 0.5:
            random = numpy.random.uniform(0.9, 1.0)
            x = numpy.array(tensorflow.image.central_crop(x, random))
            x = numpy.array(tensorflow.image.resize(x, (self.image_size, self.image_size), "bilinear"))
            y = numpy.array(tensorflow.image.central_crop(y, random))
            y = numpy.array(tensorflow.image.resize(y, (self.image_size, self.image_size), "nearest"))

        return x, y


    def __getitem__(self, index: int) -> tuple[numpy.ndarray, numpy.ndarray]:

        index_beg = index * self.batch_size
        index_end = min(index_beg + self.batch_size, len(self.xl_set))

        xl_batch = self.xl_set[index_beg:index_end]
        yl_batch = self.yl_set[index_beg:index_end]

        x_array = []
        y_array = []

        for xl, yl in zip(xl_batch, yl_batch):

            x = numpy.array(PIL.Image.open(xl).convert("RGB"))
            x = numpy.array(tensorflow.image.resize(x, (self.image_size, self.image_size), "bilinear"))

            y = numpy.array(PIL.Image.open(yl).convert("L"))
            y = numpy.array(numpy.expand_dims(y, axis=-1))
            y = numpy.array(tensorflow.image.resize(y, (self.image_size, self.image_size), "nearest"))
            y = numpy.array(numpy.squeeze(y, axis=-1))
            y = numpy.array(numpy.vectorize(self.label_to_index.get)(y))
            y = numpy.array(tensorflow.keras.utils.to_categorical(y, self.num_labels))

            if self.data_augmentation:
                x, y = self._augment(x, y)

            x_array.append(x)
            y_array.append(y)

        x_array = numpy.array(x_array)
        y_array = numpy.array(y_array)

        return x_array, y_array


    def length(self) -> int:

        return len(self.xl_set)


    def y(self) -> numpy.ndarray:

        y_array = []

        for index in range(self.__len__()):

            _, y = self.__getitem__(index)
            y_array.append(y)

        y_array = numpy.array(numpy.concatenate(y_array, axis=0))

        return y_array


    @classmethod
    def from_directory(cls, directory: pathlib.Path, image_size: int, batch_size: int, data_augmentation: bool) -> tuple["DatasetLoader", "DatasetLoader", "DatasetLoader", str, list[int]]:

        xl_all = []
        yl_all = []

        for location in directory.rglob("*"):
            if location.is_file():
                if location.parent.stem == "Images":
                    xl_all.append(location)
                if location.parent.stem == "Masks":
                    yl_all.append(location)

        xl_train, xl_temp, yl_train, yl_temp = sklearn.model_selection.train_test_split(xl_all, yl_all, train_size=0.8, stratify=None)
        xl_test, xl_val, yl_test, yl_val = sklearn.model_selection.train_test_split(xl_temp, yl_temp, train_size=0.5, stratify=None)

        title = directory.stem
        labels = set()
        for yl in yl_all: labels.update(numpy.unique(numpy.array(PIL.Image.open(yl).convert("L"))))
        labels = sorted(map(int, labels))

        dl_train = cls(xl_train, yl_train, labels, image_size, batch_size, data_augmentation)
        dl_test = cls(xl_test, yl_test, labels, image_size, batch_size, False)
        dl_val = cls(xl_val, yl_val, labels, image_size, batch_size, False)

        return dl_train, dl_test, dl_val, title, labels
