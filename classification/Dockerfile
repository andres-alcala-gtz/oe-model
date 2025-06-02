FROM tensorflow/tensorflow:2.14.0-gpu

WORKDIR /usr/local/app

COPY ./ ./

RUN pip install --no-cache-dir click==8.1.7 pandas==2.2.2 openpyxl==3.1.2 matplotlib==3.9.2 scikit-learn==1.5.1 gradio==5.31.0

EXPOSE 7860
