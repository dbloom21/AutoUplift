#%% Define Class
#import os
#os.system(f"pip install azureml-train-automl-client==1.24.0")
import azure.functions as func
from azureml.core.authentication import MsiAuthentication
#from azureml.train.automl import AutoMLConfig
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import json
import os

import requests
import time
import ast

# Import Azure ML SDK modules
#import azureml.core
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core import Experiment
# from azureml.core import Run
# from azureml.core.webservice import Webservice
# from azureml.core.image import ContainerImage
from azureml.core.webservice import AciWebservice
#from azureml.core.conda_dependencies import CondaDependencies 
from azureml.train.automl import AutoMLConfig
#import azureml.train.automl.runtime

from azureml.core import Dataset
from azureml.core import Environment
import logging
#from azureml.train.automl import AutoMLConfig
#from azureml.widgets import RunDetails
#from azureml.core.webservice import LocalWebservice, Webservice
from azureml.core.model import InferenceConfig, Model
from datetime import datetime
from azureml.train.automl.run import AutoMLRun
import azureml.pipeline.core

def main(req: func.HttpRequest) -> func.HttpResponse:
  class Uplift:
    
    def __init__(self, email_recipient, workspace, experiment, db, compute_target, valuable_col, transformation, dataset = None, json_val = None):
      #Get Workspace
      msi_auth = MsiAuthentication()
      self.ws = Workspace.get(name=workspace,
                        subscription_id='e635454b-e4d3-4bbb-abfd-5c66ca572d75', 
                        resource_group='mi2',
                        auth = msi_auth
                      )

      # Write configuration to local file
      self.ws.write_config()  
      
      # Create Azure ML Experiment
      self.exp = Experiment(workspace=self.ws, name=experiment)
      
      self.compute_target = self.ws.compute_targets[compute_target]
      
      if dataset != None:
        self.dataset = Dataset.get_by_name(self.ws, name = dataset)

      self.json = json_val
      
      # get the datastore to upload prepared data
      self.datastore = self.ws.get_default_datastore()
      
      self.local_path = "data"
      try:
        os.mkdir(self.local_path)
      except:
        pass
      #upload_file_path = os.path.join(local_path, local_file_name)

      #Get today's date format
      now = datetime.now()
      self.date = now.strftime("%m-%d-%Y")
      self.date2 = now.strftime("%m%d%Y")

      #Name of CSV files
      self.local_file_name = 'original_' + self.date + '.csv'
      self.local_file_trans = 'transformed_' + self.date + '.csv'
      self.local_file_val = 'valuable_' + self.date + '.csv'
      
      #Pre-set variables used throughout
      # self.transformation = None
      
      # self.valuable_col = []
      
      self.label = "target"
      
      self.columns = []
      
      self.data = pd.DataFrame()
      self.transform_df = pd.DataFrame()
      
      self.ds = None
      self.transform_ds = None
      self.valuable_ds = None
      
      
      
      self.automl_run = None
      self.trans_run = None
      self.val_run = None
      
      
      self.fitted_model = None
      self.fitted_model_trans = None
      self.fitted_model_val = None
      
      self.test_data = None
      self.test_data_trans = None
      self.test_data_val = None

      self.accuracy = None
      self.accuracy_trans = None
      self.accuracy_val = None

      self.best_metrics = {}
      self.best_metrics_trans = {}
      self.best_metrics_val = {}
      
      self.metrics_df = pd.DataFrame()
    
      self.best = None
    
      self.model_name = None
      #self.model_folder = 'C:/Users/derek bloom/OneDrive - AFS Technologies - AFSI/Documents/Exceedra Pro/Pro Uplift/outputs/'
      self.model = None
      
      self.X = pd.DataFrame()

      self.entry_script = "automated_entry_" + self.date2 + ".py"

      self.environment = 'AutoEnv.yml'

      self.email_recipient = email_recipient

      self.db = db

      self.client = experiment

      if self.json != None:
        all_data = pd.DataFrame(self.json)
      
      else:
        all_data = self.dataset.to_pandas_dataframe()
      

      
      all_data = all_data.loc[all_data['forecast']>0]
      
      ###Determine Target
      try:
        all_data = all_data.loc[all_data['sales']>0]
        all_data['target'] = [(A/B) for A,B in zip(all_data["sales"],all_data["forecast"])]
      except:
        pass
      #Outliers
      all_data = all_data.loc[all_data['target']>.5]
      all_data = all_data.loc[all_data['target']<15]
      all_data = all_data.loc[pd.notnull(all_data['discountPercent'])]
      
      
      self.columns = all_data.columns.drop(["promoCode","forecast"])
      
      self.data = all_data[self.columns]

      self.valuable_col = ast.literal_eval(valuable_col)
      logging.info(self.valuable_col)
      
      self.transformation = transformation
      if self.transformation != None:
        logging.info(self.data.columns)
        self.transform_df = self.data[self.valuable_col]
        self.transform_df["target"] = [np.sqrt(t) if transformation == "sqrt"
                                      else np.square(t) if transformation == 'square'
                                      else np.log(t) if transformation == 'log'
                                      else t
                                      for t in self.transform_df["target"]]
    
    
    def Transform(self,val,trans):
      #Transform uplift back to proper form
        import numpy as np
        if trans == 'log':
            val = np.e**(val)
        elif trans == 'sqrt':
            val = np.square(val)
        elif trans == 'square':
            val = np.sqrt(val)
        else:
            val = val
        return(val)
      
      
    def evaluate(self, model, test_features, test_labels,trans):
        #run model on test set to evaluate model accuracy
        import numpy as np
        import pandas as pd
    
        predictions = model.predict(test_features)
        errors = abs(self.Transform(abs(predictions),trans) - self.Transform(abs(test_labels),trans))
        mape = 100 * np.mean(errors / self.Transform(abs(test_labels),trans))
        mape2 = 100 * (errors / self.Transform(abs(test_labels),trans))
        accuracy = 100 - mape
        
        df = pd.DataFrame(dict(test_labels = test_labels))
        # Data Frame to examine test data
        df['predictions'] = self.Transform(predictions,trans)
        df['test_labels'] = [self.Transform(tl,trans) for tl in df['test_labels']]
        df['errors'] = errors
        df['mape'] = mape2
    
        return (accuracy)       
    
    def internal_run(self):
      """Try Random Forest model on multiple datasets.
        Process includes cleaning outliers, transformations on
        Target, and determining which variables bring value.
      """
      ##file = 'C:/Users/derek bloom/OneDrive - AFS Technologies - AFSI/Documents/Exceedra Pro/Pro Uplift/data/Pro Data 2-17.json'
      if self.json != None:
        all_data = pd.DataFrame(self.json)
      
      else:
        all_data = self.dataset.to_pandas_dataframe()
      

      
      all_data = all_data.loc[all_data['forecast']>0]
      
      ###Determine Target
      try:
        all_data = all_data.loc[all_data['sales']>0]
        all_data['target'] = [(A/B) for A,B in zip(all_data["sales"],all_data["forecast"])]
      except:
        pass
      #Outliers
      all_data = all_data.loc[all_data['target']>.5]
      all_data = all_data.loc[all_data['target']<15]
      all_data = all_data.loc[pd.notnull(all_data['discountPercent'])]
      
      
      self.columns = all_data.columns.drop(["promoCode","forecast"])
      
      self.data = all_data[self.columns]
      
      label = self.label
      features = self.columns.drop(label)
      
      accuracy_dict = {}
      for trans in ["sqrt", "square", "log", None]:
          transformed = all_data[self.columns]
          transformed["target"] = [np.sqrt(t) if trans == "sqrt"
                                  else np.square(t) if trans == 'square'
                                  else np.log(t) if trans == 'log'
                                  else t
                                  for t in transformed["target"]]
        
          label = "target"
          features = self.columns.drop(label)
      
      
          train_data, test_data = train_test_split(transformed, test_size=0.15, 
                                                  random_state=21)
          train_features = train_data[features]
          train_labels = train_data[label]
          test_features = test_data[features]
          test_labels = test_data[label]
      
          rf = RandomForestRegressor()
      
          rf.fit(train_features, train_labels)
      
          importance = rf.feature_importances_
      
          importance_df = pd.DataFrame(importance, features)
          importance_df.columns = ['importance']
          valuable = importance_df.query('importance >= .05')
          valuable = set(list(valuable.index) + ["cust", "sku"])
      
          accuracy = self.evaluate(rf, test_features, test_labels, trans)
          accuracy_dict[trans] = {}
          accuracy_dict[trans]["accuracy"] = accuracy
          accuracy_dict[trans]["valuable"] = valuable
      
      transformation = max(accuracy_dict, key=lambda k: accuracy_dict[k]["accuracy"])
      self.valuable_col = accuracy_dict[transformation]["valuable"]
      self.valuable_col.add(label)

      if transformation != None:
        self.transformation = transformation
        self.transform_df = self.data[self.valuable_col]
        self.transform_df["target"] = [np.sqrt(t) if transformation == "sqrt"
                                    else np.square(t) if transformation == 'square'
                                    else np.log(t) if transformation == 'log'
                                    else t
                                    for t in self.transform_df["target"]]
    
    
    def create_datasets(self):
      #Turn data into Datasets to be used in experiments
      file_path_O = os.path.join(self.local_path, self.local_file_name)
      self.data.to_csv(file_path_O, index=False)
      self.datastore.upload_files(files=[file_path_O], target_path='data')
      self.ds = Dataset.Tabular.from_delimited_files(path = [(self.datastore, ('data/original_' + self.date + '.csv'))])
      

      if self.transformation != None:
          file_path_T = os.path.join(self.local_path, self.local_file_trans)
          self.transform_df.to_csv(file_path_T, index=False)
          self.datastore.upload_files(files = [file_path_T], target_path='data')
          self.transform_ds = Dataset.Tabular.from_delimited_files(path = \
                                  [(self.datastore, ('data/transformed_' + self.date + '.csv'))])
      
      if len(self.valuable_col) < len(self.columns):
          file_path_V = os.path.join(self.local_path, self.local_file_val)
          self.data[self.valuable_col].to_csv(file_path_V, index=False)
          self.datastore.upload_files(files = [file_path_V], target_path='data')
          self.valuable_ds = Dataset.Tabular.from_delimited_files(path = [(self.datastore, ('data/valuable_' + self.date + '.csv'))])
      
      
      # upload the local file from src_dir to the target_path in datastore
      #self.datastore.upload(src_dir='CSVs', target_path='data')



    def experiment_run(self):
      #Run Experiments
      automl_settings = {
          "n_cross_validations": 3,
          "primary_metric": 'normalized_root_mean_squared_error',
          "enable_early_stopping": True, 
          "experiment_timeout_hours": 0.3, #for real scenarios we reccommend a timeout of at least one hour 
          "max_concurrent_iterations": 1,
          "max_cores_per_iteration": -1,
          "verbosity": logging.INFO,
      }

      if self.transform_ds != None:
        train_data_trans, self.test_data_trans = self.transform_ds.random_split(percentage=0.85, seed=223)
        automl_config_trans = AutoMLConfig(task = 'regression',
                                  compute_target = self.compute_target,
                                  training_data = train_data_trans,
                                  label_column_name = self.label,
                                  **automl_settings
                                  )
        self.trans_run = self.exp.submit(automl_config_trans)
        
      if self.valuable_ds != None:
        train_data_val, self.test_data_val = self.valuable_ds.random_split(percentage=0.85, seed=223)
        automl_config_val = AutoMLConfig(task = 'regression',
                                  compute_target = self.compute_target,
                                  training_data = train_data_val,
                                  label_column_name = self.label,
                                  **automl_settings
                                  )
        self.val_run = self.exp.submit(automl_config_val)


      self.email_test()

      train_data, self.test_data = self.ds.random_split(percentage=0.85, seed=223)
      automl_config = AutoMLConfig(task = 'regression',
                                  compute_target = self.compute_target,
                                  training_data = train_data,
                                  label_column_name = self.label,
                                  **automl_settings
                                  )    
      self.automl_run = self.exp.submit(automl_config)
      self.automl_run.wait_for_completion(show_output=False)

      logging.info(self.automl_run)
    
      


    def experiment_results(self):
    #   #Get Results of 
      self.run = AutoMLRun(self.exp,run_id = self.automl_run.id)
      logging.info(self.run)
      best_run, self.fitted_model = self.run.get_output()
      self.best_metrics = best_run.get_metrics()

      
      # except:
      #   logging.info("metrics")
      #   logging.info(self.run.get_metrics())
      #   logging.info("best run")
      #   best_run = self.run._get_best_child_run()
      #   logging.info(best_run)
      #   logging.info(best_run[0].id)
      #   self.best_run = AutoMLRun(self.exp,run_id = best_run[0].id)
      #   logging.info(self.best_run)
      #   #self.best_metrics = self.best_run.get_metrics()
      #   #logging.info("best metrics")
      #   #logging.info(self.best_metrics)
      #   logging.info("best model metrics")
      #   logging.info(self.best_run.get_metrics())
      #   logging.info(best_run[1])
      #   try:
      #     logging.info(self.best_run.get_output())
      #     logging.info("this worked")
      #   except:
      #     from azureml.train.automl import _model_download_utilities
      #     logging.info(_model_download_utilities._download_automl_model(self.run._get_best_child_run()[0],self.run._get_best_child_run()[1]))
      #     logging.info("this worked too")
      #best_mape = best_metrics["mean_absolute_percentage_error"]
      
      if self.trans_run != None:
        self.run_trans = AutoMLRun(self.exp,run_id = self.trans_run.id)
        best_run_trans, self.fitted_model_trans = self.run_trans.get_output()
        self.best_metrics_trans = best_run_trans.get_metrics()
        #best_mape_trans = best_metrics_trans["mean_absolute_percentage_error"]
      
      if self.val_run != None:
        self.run_val = AutoMLRun(self.exp,run_id = self.val_run.id)
        best_run_val, self.fitted_model_val = self.run_val.get_output()
        self.best_metrics_val = best_run_val.get_metrics()
        # = best_metrics_val["mean_absolute_percentage_error"]
        
      
    def metrics_get(self):
      #Get Accuracy Metrics
      metrics_df = pd.DataFrame.from_dict([self.best_metrics, self.best_metrics_trans,self.best_metrics_val])
      metrics_df = metrics_df.set_index([pd.Index(["original", "trans", "val"])])
      metrics_df['mae_rank'] = metrics_df['mean_absolute_error'].rank()
      metrics_df['mape_rank'] = metrics_df['mean_absolute_percentage_error'].rank()
      metrics_df['nrmse_rank'] = metrics_df['normalized_root_mean_squared_error'].rank()
      
      self.metrics_df = metrics_df

    def determine_accuracy(self):
      #Turn datasets to df for testing
      test_df = self.test_data.to_pandas_dataframe()
      y_test = test_df["target"]
      test_df = test_df.drop("target",1)
      ##y_pred_test = fitted_model.predict(test_df)
      ##y_residual_test = y_test - y_pred_test
      self.accuracy = self.evaluate(self.fitted_model, test_df, y_test,None)
      
      if self.test_data_trans != None:
        test_df_trans = self.test_data_trans.to_pandas_dataframe()
        y_test_trans = test_df_trans["target"]
        test_df_trans = test_df_trans.drop("target",1)
        self.accuracy_trans = self.evaluate(self.fitted_model_trans, test_df_trans, y_test_trans,self.transformation)

      if self.test_data_val != None:
        test_df_val = self.test_data_val.to_pandas_dataframe()
        y_test_val = test_df_val["target"]
        test_df_val = test_df_val.drop("target",1)
        self.accuracy_val = self.evaluate(self.fitted_model_val, test_df_val, y_test_val,None)

    def model_select(self):
      #Determine best model   
      self.metrics_df['accuracy'] = [self.accuracy,self.accuracy_trans,self.accuracy_val]
      self.metrics_df['accuracy_rank'] = self.metrics_df['accuracy'].rank(ascending=False)
      self.metrics_df["total_rank"] = self.metrics_df['mae_rank'] + self.metrics_df['mape_rank'] + self.metrics_df['nrmse_rank'] + self.metrics_df['accuracy_rank'] * 2
      
      self.best = self.metrics_df[['total_rank']].idxmin()[0]



    def model_register(self):
      # try:
      #   # Freeze the model
      #   now = datetime.now()
      #   self.model_name = self.best + "-"+ now.strftime("%m-%d-%Y")
        
        
      #   #Register Model
      if self.best == "trans":
          df = self.transform_ds.to_pandas_dataframe()
          df = df.dropna()
          self.final_accuracy = self.accuracy_trans
      #       model_best = self.fitted_model_trans
      #       y = df[self.label]
      #       X = df.drop(self.label,1)
      #       #self.fitted_model_trans.fit(X,y) 
            

        
      elif self.best == "val":
          df = self.valuable_ds.to_pandas_dataframe()
          df = df.dropna()
          self.final_accuracy = self.accuracy_val
      #       model_best = self.fitted_model_val
      #       self.transformation = None
            
      else:
          df = self.ds.to_pandas_dataframe()
          df = df.dropna()
          self.final_accuracy = self.accuracy
      #       model_best = self.fitted_model
      #       self.transformation = None
            
        
      self.X = df.drop(self.label,1)

      now = datetime.now()
      self.model_name2 = self.client + "-"+ self.db + "-" + now.strftime("%m-%d-%Y")
      logging.info(self.model_name2)

      if self.best == "trans":
        self.model = self.run_trans.register_model(model_name = self.model_name2)
          
      elif self.best == "val":
        self.model = self.run_val.register_model(model_name = self.model_name2)
          
      else:
        self.model = self.run.register_model(model_name = self.model_name2)

      self.model_name = self.model.name
      logging.info(self.model_name)
      


    def scoring_write(self):
      #writefile entry_script.py
      def trans_result(transformation):
          if transformation == 'log':
            trans_str = 'data["uplift"] = [np.e**(U) for U in data["uplift"]]'
          elif transformation == 'sqrt':
            trans_str = 'data["uplift"] = [np.square(U) for U in data["uplift"]]'
          elif transformation == 'square':
            trans_str = 'data["uplift"] = [np.sqrt(U) for U in data["uplift"]]'
          else:
            trans_str = 'data["uplift"] = data["uplift"]' 
          return(str(trans_str))
        
      str_columns = repr(list(self.X))

      if self.transformation == None:
        self.transformation = 'None'
      

      trans_str = trans_result(self.transformation)
      logging.info(trans_str)
      
      file_text = """
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

#from sklearn.linear_model import LinearRegression

from azureml.core.model import Model
  

def init():
    global model
    # retrieve the path to the model file using the model name
    model_path = Model.get_model_path('""" + self.model_name + """')
    model = joblib.load(model_path)

def run(raw_data):
    data = pd.DataFrame(json.loads(raw_data)['data'])
    # make prediction
    data = data[""" + str_columns + """]
    data["uplift"] = model.predict(data)
    """ + str(trans_str) +       """
    data = data[["cust", "sku", "uplift"]]
    result = data.to_dict('records')
    return result
  """
      logging.info(file_text)
      file = open(self.entry_script, "w")
      file.write(file_text)
      
      file.close()


    def email_results(self):
      email_body = "Your uplift model was deployed successfully. The endpoint for your model is: " + str(self.service.scoring_uri) + \
                    ". The authentication keys for your endpoint are: " + str(self.service.get_keys()) #+ \
                    #". The accuracy of your model was: " + str(self.final_accuracy)
      
      logging.info(email_body)
      
      data_json = {
        "messages": [
          {
            "body": email_body,
            "id": 7,
            "subject": "Uplift Model Deployed",
            "to": self.email_recipient
          }
        ]
      }

      r = requests.post("https://prod-114.westeurope.logic.azure.com:443/workflows/d185b822d0d3455ebd2159d5e195d94b/triggers/manual/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=3LXJvUC90CClweG1_pU7mjaoUKb6DJX124eUs2eC5Gc", 
                        json=data_json)
      
      logging.info("Request sent to " + r.url + " with status " + str(r.status_code))

    def model_deploy(self):
      #Deploy Model
      myenv = Environment.from_conda_specification('my_environment',self.environment)

      logging.info(myenv)
      #service_name = 'my-custom-env-service'
      logging.info(type(self.entry_script))
      inference_config = InferenceConfig(environment = myenv, entry_script = self.entry_script)
      logging.info(inference_config)
      aci_config = AciWebservice.deploy_configuration(cpu_cores=.1, memory_gb=.5, auth_enabled=True)
      
      self.service = Model.deploy(workspace=self.ws,
                            name=self.model_name2,
                            models=[self.model],
                            inference_config=inference_config,
                            deployment_config=aci_config,
                            overwrite=True)
      #self.service.wait_for_deployment(show_output=True)
      # time.sleep(60)
      # self.email_results()
      self.call_function()

      

      


    def run_all(self):
      self.internal_run()
      self.create_datasets()
      self.experiment_run()
      self.experiment_results()
      self.metrics_get()
      self.determine_accuracy()
      self.model_select()
      self.model_register()
      self.scoring_write()
      self.model_deploy()



    def get_model(self):
      self.best = 'original'
      self.ds = Dataset.Tabular.from_delimited_files(path = [(self.datastore, ('data/original_07-15-2021.csv'))])
      # self.run = AutoMLRun(self.exp,run_id = 'AutoML_d7a2f751-8310-41ed-8ee5-b4233579c1c3')
      # logging.info(self.run)
      # best_run, self.fitted_model = self.run.get_output()
      # self.best_metrics = best_run.get_metrics()
      self.transformation = None
      df = self.ds.to_pandas_dataframe()
      df = df.dropna()
      self.X = df.drop(self.label,1)
      now = datetime.now()
      self.model_name2 = "client1-qa-07-27-2021"#self.best + "-"+ now.strftime("%m-%d-%Y")
      #logging.info(self.model_name)
      self.model = Model(self.ws, self.model_name2)
      self.model_name = self.model.name


    def get_service(self):
      logging.info(self.model_name)
      self.service = AciWebservice(self.ws, "pro-2-5-21")
      logging.info(self.service)
      self.keys = self.service.get_keys()

    def env_write(self):
      env_text = """
  # Conda environment specification. The dependencies defined in this file will
  # be automatically provisioned for runs with userManagedDependencies=False.

  # Details about the Conda environment file format:
  # https://conda.io/docs/user-guide/tasks/manage-environments.html#create-env-file-manually

  name: project_environment
  dependencies:
    # The python interpreter version.
    # Currently Azure ML only supports 3.5.2 and later.
  - python=3.6.2

  - pip:
    - azureml-train-automl-runtime==1.22.0
    - inference-schema
    - azureml-interpret==1.22.0
    - azureml-defaults==1.22.0
  - numpy>=1.16.0,<1.19.0
  - pandas==0.25.1
  - scikit-learn==0.22.1
  - py-xgboost<=0.90
  - fbprophet==0.5
  - holidays==0.9.11
  - psutil>=5.2.2,<6.0.0
  channels:
  - anaconda
  - conda-forge
  """

      file = open(self.environment, "w")
      file.write(env_text)
      
      file.close()


    def email_test(self):
      email_body = "Experiments have begun for your uplift model. Results should be expected in approximately thirty minutes."
      data_json = {
        "messages": [
          {
            "body": email_body,
            "id": 7,
            "subject": "Uplift Update 1",
            "to": self.email_recipient
          }
        ]
      }

      r = requests.post("https://prod-114.westeurope.logic.azure.com:443/workflows/d185b822d0d3455ebd2159d5e195d94b/triggers/manual/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=3LXJvUC90CClweG1_pU7mjaoUKb6DJX124eUs2eC5Gc", 
                        json=data_json)
      
    
    def email_test2(self):
      email_body = "The accuracy of your model was: " + str(self.final_accuracy) + \
                    ". The transformation for your model was: " + self.transformation +\
                    ". The variables used for your model were: " + repr(list(self.X))
      data_json = {
        "messages": [
          {
            "body": email_body,
            "id": 7,
            "subject": "Uplift Update 2",
            "to": self.email_recipient
          }
        ]
      }

      r = requests.post("https://prod-114.westeurope.logic.azure.com:443/workflows/d185b822d0d3455ebd2159d5e195d94b/triggers/manual/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=3LXJvUC90CClweG1_pU7mjaoUKb6DJX124eUs2eC5Gc", 
                        json=data_json)

    def call_function(self):
      data_json = {
        "model": self.model.name
      }
      r = requests.post("https://docker-uplift.azurewebsites.net/api/HttpExample2?code=BC97ynLFFHzBEeND5nev8RaHXhEyAMs67pkmQdFOI8SyEkPMUheykA==",
        json=data_json)
      logging.info("Request sent to " + r.url + " with status " + str(r.status_code))
  ## %%
  try:
    req_body = req.get_json()
  except:
    req_body = req

  
  #logging.info(req_body)
  workspace = req_body.get("workspace")
  logging.info(workspace)
  client = req_body.get("client")
  logging.info(client)
  db = req_body.get("db")
  logging.info(db)
  valuable_col = req_body.get("valuable_col")
  logging.info(valuable_col)
  transformation = req_body.get("transformation")
  logging.info(transformation)
  #compute_target = req_body.get("compute_target")
  #logging.info(compute_target)
  try:
    dataset = req_body.get("dataset")
  except:
    dataset = None
  logging.info(dataset)
  try:
    json_val = req_body.get("json")
  except:
    json_val = None
  #logging.info(json)
  email_recipient = req_body.get("email_recipient")
  logging.info(email_recipient)

  logging.info("definition complete")
  uplift_run = Uplift(email_recipient, workspace, client, db, "test-2-2-21", valuable_col, transformation, dataset, json_val)    
  logging.info("uplift_run initiated")
  # uplift_run.internal_run()
  # logging.info("internal run complete")
  
  uplift_run.create_datasets()
  logging.info("datasets created")
  
  #uplift_run.email_test()

  uplift_run.experiment_run()
  logging.info("experiments ran")
  uplift_run.experiment_results()
  logging.info("experiments results gathered")
  uplift_run.metrics_get()
  logging.info("metrics returned")
  uplift_run.determine_accuracy()
  logging.info("accuracy determined")
  uplift_run.model_select()
  logging.info("model selected")
  uplift_run.model_register()
  logging.info("model registered")

  uplift_run.email_test2()

  uplift_run.scoring_write()
  logging.info("scoring file written")
  uplift_run.env_write()
  logging.info("environment written")
  uplift_run.model_deploy()
  logging.info("model deployed")
  
  # time.sleep(60)
  # uplift_run.get_service()
  # logging.info("service received") 
  # uplift_run.email_results()
  # logging.info("results emailed with delay")

  uplift_run.call_function()
  logging.info("function called")
