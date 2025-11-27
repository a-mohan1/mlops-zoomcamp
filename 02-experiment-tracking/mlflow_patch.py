# mlflow_patch.py
# This issue was fixed in newer MLflow versions. Check your version and upgrade.

import os
import boto3
import mlflow.store.artifact.s3_artifact_repo as s3_repo

def patch_mlflow_for_iam_role():
    """Patch MLflow to use IAM role instead of explicit credentials"""
    
    # Clear any partial credentials
    for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN']:
        os.environ.pop(key, None)
    
    # Set region
    if 'AWS_DEFAULT_REGION' not in os.environ:
        os.environ['AWS_DEFAULT_REGION'] = 'ap-southeast-2'
    
    # Patch the S3 client creation
    def patched_get_s3_client(*args, **kwargs):
        """Force boto3 to use IAM role"""
        return boto3.client('s3', region_name=os.environ.get('AWS_DEFAULT_REGION'))
    
    s3_repo._get_s3_client = patched_get_s3_client
    print("✓ MLflow patched to use IAM role for S3 access")