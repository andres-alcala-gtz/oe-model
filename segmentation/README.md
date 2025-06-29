# Optimized Ensembled Model - Classification


## 1. Preparing the dataset

Place the dataset within the `classification/` directory following the structure outlined below:

```text
classification/
└─ dataset/
   ├─ label_1/
   │  ├─ image_1.png
   │  ├─ image_2.jpg
   │  └─ image_n.png
   ├─ label_2/
   │  ├─ image_1.png
   │  ├─ image_2.jpg
   │  └─ image_n.png
   └─ label_n/
      ├─ image_1.png
      ├─ image_2.jpg
      └─ image_n.png
```


## 2. Adjusting the environment

Modify the `environment.py` file as required. The default configuration is as follows:

```python
IMAGE_SIZE = 512
BATCH_SIZE = 16
EXPERIMENT_RUNS = 10
FIGURE_SIZE = 8.5
DATA_AUGMENTATION = True
```


## 3. Navigating to the working directory

Navigate to the project directory:

```bash
cd classification
```


## 4. Building the docker image

Build the docker image with the following command:

```bash
docker build -t classification .
```


## 5. Running the docker container

Run the container using:

```bash
docker run -it --rm --gpus all -v "$(pwd):/usr/local/app" -p 7860:7860 classification
```


## 6A. Training a single model

Train and launch a model:

```bash
python main.py
```


## 6B. Training multiple experimental models

Train multiple models for experimentation and benchmarking:

```bash
python main_experimentation.py
```
