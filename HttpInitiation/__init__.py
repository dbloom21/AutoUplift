#%% Define Class
#import os
#os.system(f"pip install azureml-train-automl-client==1.24.0")
import azure.functions as func
from azureml.core.authentication import MsiAuthentication
#from azureml.train.automl import AutoMLConfig
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import json
import os

import requests
import time

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
    
    def __init__(self, email_recipient, workspace, experiment, db, compute_target, dataset = None, json_val = None):
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
      self.transformation = None
      
      self.valuable_col = []
      
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

      #if transformation != None:
      self.transformation = transformation
        # self.transform_df = self.data[self.valuable_col]
        # self.transform_df["target"] = [np.sqrt(t) if transformation == "sqrt"
        #                             else np.square(t) if transformation == 'square'
        #                             else np.log(t) if transformation == 'log'
        #                             else t
        #                             for t in self.transform_df["target"]]

    
    
    def call_function(self):
      data_json = {"email_recipient": "derek.bloom@afsi.com", "workspace":"ExceedraPro", "client":"client1", "db": "qa",\
                   "dataset":"Auto-2-4-21", "transformation": str(self.transformation), "valuable_col": repr(list(self.valuable_col))}
      logging.info(data_json)
      
      try:
        r = requests.post("https://docker-uplift.azurewebsites.net/api/HttpExample?code=hUKo8ORMC5CFgqqfDgkzu1RtEMTcoa/VISkTZ/WOn4uixxjtk0clCA==",\
        json=data_json, timeout=60)
        logging.info("Request sent to " + r.url + " with status " + str(r.status_code))
      except requests.exceptions.ReadTimeout: 
        pass
      
      
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
  uplift_run = Uplift(email_recipient, workspace, client, db, "test-2-2-21", dataset, json_val)    
  logging.info("uplift_run initiated")
  uplift_run.internal_run()
  logging.info("internal run complete")
  
  
  uplift_run.call_function()
  logging.info("function called")
 
  Output = "Uplift Model Initiated. Useful variables are " + repr(list(uplift_run.valuable_col)) + \
            ". Possible transformation is " + uplift_run.transformation + "."
  return func.HttpResponse(body=str(Output), mimetype="application/json")
