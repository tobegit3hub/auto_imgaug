#!/usr/bin/env python

import json

from advisor_client.client import AdvisorClient


def main(train_function):
  client = AdvisorClient()

  # Get or create the study
  study_configuration = {
      "goal":
      "MINIMIZE",
      "randomInitTrials":
      1,
      "maxTrials":
      5,
      "maxParallelTrials":
      1,
      "params": [{
          "parameterName":
          "operation_name1",
          "type":
          "CATEGORICAL",
          "minValue":
          0,
          "maxValue":
          0,
          "feasiblePoints":
          "Fliplr, Crop, GaussianBlur, ContrastNormalization, AdditiveGaussianNoise, Multiply",
          "scallingType":
          "LINEAR"
      }, {
          "parameterName": "magnitude1",
          "type": "INTEGER",
          "minValue": 1,
          "maxValue": 11,
          "scallingType": "LINEAR"
      }, {
          "parameterName": "probability1",
          "type": "INTEGER",
          "minValue": 1,
          "maxValue": 10,
          "scallingType": "LINEAR"
      }, {
          "parameterName":
          "operation_name2",
          "type":
          "CATEGORICAL",
          "minValue":
          0,
          "maxValue":
          0,
          "feasiblePoints":
          "Fliplr, Crop, GaussianBlur, ContrastNormalization, AdditiveGaussianNoise, Multiply",
          "scallingType":
          "LINEAR"
      }, {
          "parameterName": "magnitude2",
          "type": "INTEGER",
          "minValue": 1,
          "maxValue": 11,
          "scallingType": "LINEAR"
      }, {
          "parameterName": "probability2",
          "type": "INTEGER",
          "minValue": 1,
          "maxValue": 10,
          "scallingType": "LINEAR"
      }, {
          "parameterName":
          "operation_name3",
          "type":
          "CATEGORICAL",
          "minValue":
          0,
          "maxValue":
          0,
          "feasiblePoints":
          "Fliplr, Crop, GaussianBlur, ContrastNormalization, AdditiveGaussianNoise, Multiply",
          "scallingType":
          "LINEAR"
      }, {
          "parameterName": "magnitude3",
          "type": "INTEGER",
          "minValue": 1,
          "maxValue": 11,
          "scallingType": "LINEAR"
      }, {
          "parameterName": "probability3",
          "type": "INTEGER",
          "minValue": 1,
          "maxValue": 10,
          "scallingType": "LINEAR"
      }]
  }
  #study = client.create_study("Study", study_configuration, "BayesianOptimization")
  study = client.create_study("Study", study_configuration,
                              "RandomSearchAlgorithm")
  #study = client.get_study_by_id(6)

  # Get suggested trials
  trials = client.get_suggestions(study.id, 3)

  #import ipdb;ipdb.set_trace()

  # Generate parameters
  parameter_value_dicts = []
  for trial in trials:
    parameter_value_dict = json.loads(trial.parameter_values)
    print("The suggested parameters: {}".format(parameter_value_dict))
    parameter_value_dicts.append(parameter_value_dict)

  # Run training
  metrics = []
  for i in range(len(trials)):
    metric = train_function(**parameter_value_dicts[i])
    #metric = train_function(parameter_value_dicts[i])
    metrics.append(metric)

    trial = trials[i]
    # Complete the trial
    client.complete_trial_with_one_metric(trial, metrics[i])

  is_done = client.is_study_done(study.id)
  best_trial = client.get_best_trial(study.id)
  print("The study: {}, best trial: {}".format(study, best_trial))


if __name__ == "__main__":
  main()
