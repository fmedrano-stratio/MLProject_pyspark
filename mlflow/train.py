# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

""" Basic packages """
import argparse
import sys
import pandas as pd
import numpy as np
seed=123

""" Rocket packages """
from rocket.mlflow import RocketUtils
from mlflow.models import infer_signature

""" Pyspark packages """
from pyspark.sql import * 
from pyspark.sql.functions import * 
from pyspark.sql.types import * 
import pyspark.version 
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier



""" ML Flow packages """
import mlflow
import mlflow.pyfunc
import mlflow.sklearn        
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec                     

""" Arguments """
def parse_args():

    parser = argparse.ArgumentParser(description='example-model')
    parser.add_argument('--training-data', type=str, help='training data set in csv')
    parser.add_argument('--with-mean', type=bool, help='StandardScaler with mean')
    parser.add_argument('--with-std', type=bool, help='StandardScaler with std')
    parser.add_argument('--max-depth', type=int, help='maximum depth of the tree')
    parser.add_argument('--min-samples-leaf', type=int, help='minimum number of samples required to be at a leaf node')
    parser.add_argument('--min-samples-split', type=int, help='minimum number of samples required to split an internal node')
    parser.add_argument('--conda-env', type=str, default='conda.yaml',help='the path to a conda environment yaml file (default: None)')

    return parser.parse_args()

# Model Evaluation
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator



""" Main """
def main():
    
    """ Data """
    #New R4 functionality 1: Direct access to Data
    
    df_train= RocketUtils.get_train_spark_dataframe()
    df_train_count = df_train.count()
    df_validate= RocketUtils.get_eval_spark_dataframe()
    df_validate_count = df_validate.count()
    
    #New R4 functionality 2: Report Logs
    RocketUtils.logInfo(f"-------ROCKET-------> Cuantos df_train rows tenemos: {df_train_count}")
    RocketUtils.logInfo(f"-------ROCKET-------> Cuantos df_validate rows tenemos: {df_validate_count}")


    #New R4 functionality 3: Metadata access: Features/Target
    features= RocketUtils.get_features()
    target=RocketUtils.get_target()
    RocketUtils.logInfo(f"-------ROCKET-------> Feature types: {type(features)}")
    RocketUtils.logInfo(f"-------ROCKET-------> Target type: {type(target)}")

    # Create Data preprocessing pipeline steps
    string_columns = ["acquisition_channel","gender","marital_status","payment_method"]
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index",handleInvalid = "skip") for column in string_columns]
    input_cols = [indexer.getOutputCol() for indexer in indexers]
    output_cols_ohe = [indexer.getOutputCol()+"_OHE" for indexer in indexers]
    encoders = OneHotEncoder(inputCols=input_cols, outputCols=output_cols_ohe) 
    numerical_columns = list(set(set(df_train.columns) - set(string_columns) - set(["ingestion_date"]) - set(["renewed"])))
    # Vector assemble and Standard Scale
    continuous_feature_assembler = VectorAssembler(
        inputCols=numerical_columns,
        outputCol="unscaled_numerical_features"
    )

    continuous_feature_scaler = StandardScaler(
        inputCol="unscaled_numerical_features",
        outputCol="scaled_numerical_features",
        withStd=True,
        withMean=False
    )

    featuresColsrf = output_cols_ohe
    featuresColsrf.append("scaled_numerical_features")
    featuresAssemblerf = VectorAssembler(inputCols=featuresColsrf, outputCol="features_rf")
    # Merge all the steps
    estimators = indexers + [encoders] + [continuous_feature_assembler, continuous_feature_scaler,featuresAssemblerf]
    
    with mlflow.start_run() as run:
        #Model
        rf = RandomForestClassifier(labelCol="renewed", featuresCol="features_rf")

        # Evaluate model
        rfevaluator = BinaryClassificationEvaluator(labelCol = "renewed")

        # Create the pipeline
        estimators.append(rf)
        pipeline = Pipeline(stages=estimators)
        rfModel = pipeline.fit(df_train)
        # Signature 
        model_signature = infer_signature(df_train.drop(target), rfModel.transform(df_validate.limit(1)).select("renewed")) # coge la signatura como el tipo de output/inpt que espera
        mlflow.spark.log_model(rfModel, 'model', signature=model_signature)
        print('Model logged in run {}'.format(run.info.run_uuid))



if __name__ == '__main__':
    main()

        





