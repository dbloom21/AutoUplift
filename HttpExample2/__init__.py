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
  class Uplift_Results:
    
    def __init__(self,model, email_recipient, workspace, experiment, db, compute_target):
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
      
      # if dataset != None:
      #   self.dataset = Dataset.get_by_name(self.ws, name = dataset)

      #self.json = json
      
      # get the datastore to upload prepared data
      self.datastore = self.ws.get_default_datastore()
      
      # self.local_path = "data"
      # try:
      #   os.mkdir(self.local_path)
      # except:
      #   pass
      # #upload_file_path = os.path.join(local_path, local_file_name)

      # #Get today's date format
      # now = datetime.now()
      # self.date = now.strftime("%m-%d-%Y")
      # self.date2 = now.strftime("%m%d%Y")

      # #Name of CSV files
      # self.local_file_name = 'original_' + self.date + '.csv'
      # self.local_file_trans = 'transformed_' + self.date + '.csv'
      # self.local_file_val = 'valuable_' + self.date + '.csv'
      
      # #Pre-set variables used throughout
      # self.transformation = None
      
      # self.valuable_col = []
      
      # self.label = "target"
      
      # self.columns = []
      
      # self.data = pd.DataFrame()
      # self.transform_df = pd.DataFrame()
      
      # self.ds = None
      # self.transform_ds = None
      # self.valuable_ds = None
      
      
      
      # self.automl_run = None
      # self.trans_run = None
      # self.val_run = None
      
      
      # self.fitted_model = None
      # self.fitted_model_trans = None
      # self.fitted_model_val = None
      
      # self.test_data = None
      # self.test_data_trans = None
      # self.test_data_val = None

      # self.accuracy = None
      # self.accuracy_trans = None
      # self.accuracy_val = None

      # self.best_metrics = {}
      # self.best_metrics_trans = {}
      # self.best_metrics_val = {}
      
      # self.metrics_df = pd.DataFrame()
    
      # self.best = None
    
      # self.model_name = None
      # #self.model_folder = 'C:/Users/derek bloom/OneDrive - AFS Technologies - AFSI/Documents/Exceedra Pro/Pro Uplift/outputs/'
      # self.model = None
      
      # self.X = pd.DataFrame()

      # self.entry_script = "automated_entry_" + self.date2 + ".py"

      # self.environment = 'AutoEnv.yml'

      self.email_recipient = email_recipient

      self.db = db

      self.client = experiment
      time.sleep(240)
      self.service = AciWebservice(self.ws, model)
    


    def email_results(self):
      email_body = "Your uplift model was deployed successfully. The endpoint for your model is: " + str(self.url) + \
                    ". The authentication keys for your endpoint are: " + str(self.keys) #+ \
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



    def email_test(self):
      email_body = "Your model has been deployed. Details will be sent shortly."
      data_json = {
        "messages": [
          {
            "body": email_body,
            "id": 7,
            "subject": "Uplift Update 3",
            "to": self.email_recipient
          }
        ]
      }

      r = requests.post("https://prod-114.westeurope.logic.azure.com:443/workflows/d185b822d0d3455ebd2159d5e195d94b/triggers/manual/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=3LXJvUC90CClweG1_pU7mjaoUKb6DJX124eUs2eC5Gc", 
                        json=data_json)

  try:
    req_body = req.get_json()
  except:
    req_body = req

  
  #logging.info(req_body)
  model = req_body.get("model")

  logging.info("definition complete")
  uplift_run = Uplift_Results(model, "derek.bloom@afsi.com", "ExceedraPro", "client1", "qa", "test-2-2-21")    
  logging.info("uplift_run initiated")
  
  uplift_run.email_test()

  logging.info(uplift_run.service)
  uplift_run.url = uplift_run.service.scoring_uri
  logging.info(uplift_run.url)
  uplift_run.keys = uplift_run.service.get_keys()
  logging.info(uplift_run.keys)
 
  
  uplift_run.email_results()
  logging.info("results emailed with delay")

  
