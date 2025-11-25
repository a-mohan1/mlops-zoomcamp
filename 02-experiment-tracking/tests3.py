import boto3
s3 = boto3.client("s3")
s3.put_object(Bucket="mlflow-artifacts-remote31", Key="test.txt", Body=b"hello")