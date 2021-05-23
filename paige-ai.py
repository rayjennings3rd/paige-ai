import os
import datetime
import json
import logging
from decimal import Decimal
from enum import Enum
from hashlib import blake2b
import stringcase
import numpy as np
import pandas as pd
import boto3
from botocore.exceptions import ClientError


"""
To Run:

- boto3 will read environment variables for aws credentials - it is assumed they are set or in .aws/config

- The specified AWS S3 bucket must already be created and the aws credentials must have read and write access to it

- DynamoDB table must already be created and have a primary key of 'uuid' and aws credentials must have write access

- Three environment variables must be set: HASH_KEY, S3_BUCKET, DYNAMODB_TABLE. If any are not set the program
  will exit.

Examples for environment varibles:

export HASH_KEY='123456789123456789'
export S3_BUCKET='paigeai'
export DYNAMODB_TABLE='glucose'

- There should be a file in the specified s3 bucket with a name format like: 2021-05-18_patient_data.csv

- It is assumed that the file was uploaded to S3 on the same day but before this ETL script is run
  If the policy is that the data is from the previous day - then the code needs to be adjusted to get 
  the previous date, etc.

(Please note that I renamed the file given to me by Darren to use the current date so that it would work as designed.)

"""


#
# Read all 3 env vars
#
hash_key = os.environ['HASH_KEY'].encode('utf-8')
if hash_key is None:
    print("HASH_KEY env variable is not set")
    quit()
    
# s3_bucket should be something like 'paigeai' not a full url
s3_bucket = os.environ['S3_BUCKET']
if s3_bucket is None:
    print("S3_BUCKET env variable is not set")
    quit()

# The dynamodb table name should be something like 'glucose'
# It must already be setup in aws (with a primary key of 'uuid')
dynamodb_table = os.environ['DYNAMODB_TABLE']
if dynamodb_table is None:
    print("DYNAMODB_TABLE env variable is not set")
    quit()


class BloodGlucoseLevel(Enum):
    """ Enum class for diabetes level """
    normal = 0
    pre_diabetic = 1
    diabetic = 2


def csv_to_df(csv_file):
    """ Read CSV file at file location (which can also be a URL) on S3 """
    d = np.genfromtxt(csv_file, delimiter=',', skip_header=0, names=True, encoding='ISO-8859-1', autostrip=True,
                      dtype=None)
    return pd.DataFrame(d).infer_objects()


def get_hash(row):
    """
    Compute a hash value from the PII

    Only use patient MRN (id), name and email for the hash - do not use address since addresses frequently change
    email addresses could change as well, same with patient's last_name
    Consider using the patient id only for the hash if the above values are not static for the time range of the
    tests.
    """
    h = blake2b(key=hash_key, digest_size=16)
    h.update((str(row['patient_id']) + str(row['first_name']) + str(row['last_name']) + str(row['email'])).encode())
    return h.hexdigest()


def set_blood_sugar_level(row):
    """ Set the Enum value for the glucose level """
    if row['glucose_average'] <= 140.0:
        return BloodGlucoseLevel.normal.value  # 0
    elif row['glucose_average'] < 199.0:
        return BloodGlucoseLevel.pre_diabetic.value  # 1
    return BloodGlucoseLevel.diabetic.value  # 2


def upload_file_to_s3(file_name, bucket):
    """
    Upload a file to S3 bucket

    I assume that if the original data was allowed in S3 (with all PII) then the rows with missing data
    (no average computed) can go back to S3 for next day processing and no PII violation will happen. But
    to play it safe - the uploaded file will keep the hash uuid that will be used to update the current day's
    dataframe
    """
    s3 = boto3.client('s3')
    try:
        s3.upload_file(file_name, bucket, os.path.basename(file_name))
    except ClientError as e:
        logging.error(e)
        return False
    return True


# Get today's date to create the filename to read from s3 bucket
today = datetime.datetime.today()
date_str = today.strftime('%Y-%m-%d')

# Create the url of the file on s3
csv_file = 'https://' + s3_bucket + '.s3.amazonaws.com/' + date_str + '_patient_data.csv'

# Read the file from s3 and convert to pandas dataframe
df = csv_to_df(csv_file)

# Add a date column to the dataframe for when saving to the database
df['date'] = date_str

# Clean-up the column names and convert all to snake_case for consistency
df.columns = df.columns.str.strip()
df = df.rename(columns=stringcase.snakecase)

# TODO: What do we do about the inf glucose values? If they were given as inf I assume that they will not be
# replaced in the following day's csv. The could be converted to NaN or use a very large glucose value.
# Let's use a large value for now otherwise they may never get processed at all. Alternatively they could be
# dropped if missing some rows is acceptable.
df['glucose_mgdl_t1'] = df['glucose_mgdl_t1'].replace(np.inf, 2656.0)
df['glucose_mgdl_t2'] = df['glucose_mgdl_t2'].replace(np.inf, 2656.0)
df['glucose_mgdl_t3'] = df['glucose_mgdl_t3'].replace(np.inf, 2656.0)


# Force the glucose columns to numeric type since some have string values 'n/a'
#   'n/a' values will get converted to NaN using the coerce option
df['glucose_mgdl_t1'] = pd.to_numeric(df['glucose_mgdl_t1'], errors='coerce')
df['glucose_mgdl_t2'] = pd.to_numeric(df['glucose_mgdl_t2'], errors='coerce')
df['glucose_mgdl_t3'] = pd.to_numeric(df['glucose_mgdl_t3'], errors='coerce')

df['glucose_mgdl_t1'] = df['glucose_mgdl_t1'].astype(float)
df['glucose_mgdl_t2'] = df['glucose_mgdl_t2'].astype(float)
df['glucose_mgdl_t3'] = df['glucose_mgdl_t3'].astype(float)

#
# Create a hash of the PII data
#
df['uuid'] = df.apply(get_hash, axis=1)

# Drop PII columns - they are no longer needed and cannot be stored any further due to restrictions
df.drop(['patient_id', 'first_name', 'last_name', 'email', 'address'], axis=1, inplace=True)

# Merge in yesterday's df that had missing values by using the dataframe update method
try:
    df_missing_values = csv_to_df(s3_bucket + '/' + 'missing_averages.csv')
    if df_missing_values is not None:
        df_missing_values.set_index('uuid')
        df.update(df_missing_values)
except Exception:
    pass

# Compute glucose averages, NaN will result if any of the three values are missing
df['glucose_average'] = (df['glucose_mgdl_t1'] + df['glucose_mgdl_t2'] + df['glucose_mgdl_t3']) / 3

# Create a new dataframe with the rows where the average glucose is NaN
# That is, the average was not able to be computed due to missing values
# These rows will be put into a file and uploaded to s3 to be processed tomorrow.
df_no_average = df[df['glucose_average'].isnull()]
df_no_average.reset_index(drop=True, inplace=True)

# Remove the same rows (where average is NaN) from the primary dataframe
df.drop(df.loc[df['glucose_average'].isnull()].index, inplace=True)
df.reset_index(drop=True, inplace=True)

# Set the blood sugar level - use an int (enum) value to keep the column small (don't use a string)
df['blood_sugar_level'] = df.apply(set_blood_sugar_level, axis=1)

# Drop default index and set uuid as the index for inserting into dynamodb
df.set_index('uuid')

# Save the dataframe where the averages were unable to be computed to a csv file
df_no_average.to_csv('./missing_averages.csv', sep=',', index=False)

# Upload missing average csv to S3 to process on tomorrow's run
upload_file_to_s3('./missing_averages.csv', s3_bucket)

# Write DF to AWS DynamoDB
#   Could use RDS, or some other data store - I chose DynamoDB given the high terabyte data size and that
#   S3 is in use (ie. an AWS account exists.)
#
# DynamoDB table MUST be created with primary key set to: 'uuid'
#
# TODO: Consider a dynamodb batch write approach from the pandas dataframe for faster execution
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(dynamodb_table)

df_len = len(df)
for i in range(0, df_len):
    rec = df.to_dict('records')[i]
    rec_json = json.loads(json.dumps(rec), parse_float=Decimal)
    table.put_item(Item=rec_json)
