FROM gcr.io/ml-pipeline/ml-pipeline-local-confusion-matrix:6554e133dd453c62aea05ebb57a04f897c11d070
RUN pip install minio==6.0.2
COPY confusion_matrix_minio.py /ml/confusion_matrix_minio.py
COPY roc_minio.py /ml/roc_minio.py
ENTRYPOINT ["/bin/bash"]
