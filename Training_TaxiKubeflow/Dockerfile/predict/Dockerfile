FROM gcr.io/ml-pipeline/ml-pipeline-dataflow-tf-predict:57d9f7f1cfd458e945d297957621716062d89a49
RUN pip install minio==6.0.2
COPY  predict_minio.py /ml/predict_minio.py
ENTRYPOINT ["/bin/bash"]
