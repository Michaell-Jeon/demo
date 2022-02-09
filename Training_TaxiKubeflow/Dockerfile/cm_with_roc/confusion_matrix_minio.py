# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# A program to generate confusion matrix data out of prediction results.
# Usage:
# python confusion_matrix.py  \
#   --predictions=gs://bradley-playground/sfpd/predictions/part-* \
#   --output=gs://bradley-playground/sfpd/cm/ \
#   --target=resolution \
#   --analysis=gs://bradley-playground/sfpd/analysis \


import argparse
import json
import os
import urlparse, sys
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.python.lib.io import file_io
from minio import Minio
minio_endpoint = os.getenv('MINIO_ENDPOINT')
secretkey = os.getenv('MINIO_SECRET_KEY')
accesskey = os.getenv('MINIO_ACCESS_KEY')
envlist = [minio_endpoint, secretkey, accesskey]
if None in envlist or "" in envlist:
    print("Please Provide all these env: {},{},{}".format("MINIO_ENDPOINT","MINIO_SECRET_KEY","MINIO_ACCESS_KEY"))
    sys.exit(1)
BUCKET_NAME="visualize"


def upload_to_minio(file_to_upload,bucket_path):
    client = Minio(minio_endpoint, accesskey, secretkey,secure=False)

    if client.bucket_exists(BUCKET_NAME):
        print("Bucket {} Already Exists.".format(BUCKET_NAME))
    else:
        client.make_bucket(BUCKET_NAME)

    client.fput_object(
        BUCKET_NAME, bucket_path + '/' + file_to_upload.split('/')[-1], file_to_upload,
    )

def main(argv=None):
    parser = argparse.ArgumentParser(description='ML Trainer')
    parser.add_argument('--predictions', type=str, help='GCS path of prediction file pattern.')
    parser.add_argument('--output', type=str, help='GCS path of the output directory.')
    parser.add_argument('--target_lambda', type=str,
                      help='a lambda function as a string to compute target.' +
                           'For example, "lambda x: x[\'a\'] + x[\'b\']"' +
                           'If not set, the input must include a "target" column.')

    args = parser.parse_args()
    storage_service_scheme = urlparse.urlparse(args.output).scheme
    on_cloud = True if storage_service_scheme else False
    if not on_cloud and not os.path.exists(args.output):
        os.makedirs(args.output)
    schema_file = os.path.join(os.path.dirname(args.predictions), 'schema.json')
    schema = json.loads(file_io.read_file_to_string(schema_file))
    names = [x['name'] for x in schema]
    dfs = []
    files = file_io.get_matching_files(args.predictions)
    for file in files:
        with file_io.FileIO(file, 'r') as f:
            dfs.append(pd.read_csv(f, names=names))
    df = pd.concat(dfs)
    if args.target_lambda:
        df['target'] = df.apply(eval(args.target_lambda), axis=1)

    vocab = list(df['target'].unique())
    cm = confusion_matrix(df['target'], df['predicted'], labels=vocab)
    data = []
    for target_index, target_row in enumerate(cm):
        for predicted_index, count in enumerate(target_row):
            data.append((vocab[target_index], vocab[predicted_index], count))
    df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
    cm_file = os.path.join(args.output, 'confusion_matrix.csv')
    with file_io.FileIO(cm_file, 'w') as f:
        df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)

    BUCKET_PATH = "/".join(args.output.split('/')[3:])

    upload_to_minio(cm_file, BUCKET_PATH)

    metadata = {
        'outputs' : [{
      'type': 'confusion_matrix',
      'storage': 'minio',
      'format': 'csv',
      'schema': [
        {'name': 'target', 'type': 'CATEGORY'},
        {'name': 'predicted', 'type': 'CATEGORY'},
        {'name': 'count', 'type': 'NUMBER'},
      ],
      'source': "{}://{}/{}/{}".format("minio",BUCKET_NAME,BUCKET_PATH,cm_file.split('/')[-1]),
      # Convert vocab to string because for bealean values we want "True|False" to match csv data.
      'labels': list(map(str, vocab)),
        }]
    }

    with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)
    accuracy = accuracy_score(df['target'], df['predicted'])
    metrics = {
        'metrics': [{
      'name': 'accuracy-score',
      'numberValue':  accuracy,
      'format': "PERCENTAGE",
        }]
    }

    with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)

if __name__== "__main__":
    main()

