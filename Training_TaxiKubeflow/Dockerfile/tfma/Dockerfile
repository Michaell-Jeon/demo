FROM gcr.io/ml-pipeline/ml-pipeline-dataflow-tfma:57d9f7f1cfd458e945d297957621716062d89a49
RUN pip install minio==6.0.2
COPY model_analysis_minio.py /ml/model_analysis_minio.py
ENTRYPOINT ["/bin/bash"]
