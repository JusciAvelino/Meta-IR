# -*- coding: utf-8 -*-
"""Meta-IR.ipynb"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score)
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import LeaveOneOut

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, SignatureTranslatedAnonymousPackage
from urllib.request import urlopen as urlopen
import json
from rpy2.robjects import default_converter, pandas2ri
from rpy2.robjects.conversion import Converter, localconverter
import rpy2.robjects.numpy2ri
import itertools as it
from glob import glob
import smogn
import resreg
from xgboost import XGBRegressor

# Activate converters and filters
rpy2.robjects.numpy2ri.activate()
pandas2ri.activate()

from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline

class META_IR():

    """
    A class for meta-learning in information retrieval.

    Parameters
    ----------
    m : pandas.DataFrame
        The dataset.

    Attributes
    ----------
    m : pandas.DataFrame
        The meta-base dataset.

    Methods
    -------
    meta_feature_extraction(self)
        Extract Meta-features.
    meta_target_definition(self)
        Define meta-targets.
    balance(self, train, strategy, c)
        Data balancing.
    install_rpackages
        Install R packages
    scores(y, y_test, y_pred)
        Regression Evaluation
    repeatedKfold(self, X, y, dataset, n_splits=10, n_repeats=2, random_state=42, pipeline=None, param_grid=None) :
        Evaluation pipelines
    pipe_generation(self)
        Pipeline Generation
    select_best(self, df)
        Selects the best pipeline for the dataset
    evalutation(y_true, y_pred)
        Evaluate the performance of the model.
    train(x_train, y_train)
        Train the meta-model.
    prediction(model, x_test)
        Make predictions using the meta-model.
    generation(X, Y)
        Generate predictions using leave-one-out cross-validation.
    independent_training(self)
        Train and evaluate the model independently for each label.
    model_first(self)
        Train the model first, then the strategy.
    strategy_first(self)
        Train the strategy first, then the model.
    """

    def __init__(self, data_sets):
      self.data_sets = data_sets

    def meta_feature_extraction(self):

      string = """

      ecol <- function(){

        install.packages("devtools")
        install.packages("ECoL")
        library("devtools")
        library("ECoL")

        install.packages("UBL")
        library(UBL)


        data_sets <- Sys.glob(file.path("/ds/*.csv"))
        result_list <- list()  # Lista para armazenar os temp_df


        for(i in data_sets){
            start.time <- Sys.time()
            print(i)
            ds <- read.csv(i)
            ds_n <- basename(i)

            l <- linearity(target~ ., ds, summary=c("mean", "min", "max", "sd"))
            d <- dimensionality(target~ ., ds, summary=c("mean", "min", "max", "sd"))
            c <- correlation(target~ ., ds, summary=c("mean", "min", "max", "sd"))
            s <- smoothness(target~ ., ds, summary=c("mean", "min", "max", "sd"))

            y <- ds$target

            if (sum(is.na(y) == 0)){
              pc <- UBL::phi.control(y)
              y.phi <- phi(y, pc)

              n_raro <- sum(y.phi>0.8)
              n_row <- nrow(ds)
              n_col <- ncol(ds)-1
              p_raro <- ((n_raro/n_row)*100)
            }

            myList <- list(n_raro=n_raro, n_row=n_row, n_col=n_col, p_raro=p_raro, l=l, d=d, c=c, s=s)
            temp_df <- data.frame(matrix(unlist(myList), nrow = 1), stringsAsFactors = FALSE)

            temp_df <- data.frame(matrix(unlist(myList), nrow = 1, byrow = TRUE), stringsAsFactors = FALSE)

            result_list[[length(result_list) + 1]] <- temp_df
        }

        # Concatenar todos os dataframes da lista em um único dataframe
        result_df <- do.call(rbind, result_list)

        return(result_df)
      }

      """
      powerpack = SignatureTranslatedAnonymousPackage(string, "powerpack")
      df = powerpack.ecol()

      return df

    def meta_target_definition(self):

      pipes_params = self.pipe_generation()

      all_results = []
      for i, dataset in enumerate([data_sets[0]]):

        ds = pd.read_csv(dataset)
        path = dataset
        head, tail = os.path.split(path)
        print("=====================")
        print(path)

        X = ds.drop([ds.columns[0]], axis = 1)
        y = ds[ds.columns[0]]

        X = X.to_numpy()
        y = y.to_numpy()

        models_results = []
        for j in pipes_params:

          pipeline, param_grid = j[0], j[1]
          print(str(pipeline.steps[0][1]).split('(')[0])
          models_results.append(self.repeatedKfold(X=X, y=y, dataset=dataset, pipeline=pipeline, param_grid=param_grid))
        models_results = pd.concat(models_results).reset_index(drop=True)
        best_results = self.select_best(models_results)
        all_results.append(best_results)
      all_results = pd.concat(all_results)
      return all_results

    def balance(self, train, strategy, c):

      if strategy == "GN":
        train = iblr.gn(data = train, y = train.columns[0], samp_method=c[0], pert=c[1],  rel_thres = 0.8)
      elif strategy == "RO":
        train = iblr.ro(data = train, y = train.columns[0], samp_method=c[0], rel_thres = 0.8)
      elif strategy == "RU":
        train = iblr.random_under(data = train, y = train.columns[0], samp_method=c[0], rel_thres = 0.8)
      elif strategy == "SG":
        train =  train.dropna()
        train = smogn.smoter(data = train, y = train.columns[0], samp_method=c[0], rel_xtrm_type = 'high', rel_thres = 0.8)
        train =  train.dropna()
      elif strategy == "SMT":
        train = iblr.smote(data = train, y = train.columns[0], samp_method=c[0], rel_thres = 0.8)
      elif strategy == "WC":
        X_train = train.drop([train.columns[0]], axis = 1)
        y_train  = train[train.columns[0]]
        relevance = resreg.pdf_relevance(y_train)
        X_wercs, y_wercs = resreg.wercs(X_train, y_train, relevance, over=c[0], under=c[1])
        trainWC = np.column_stack((y_wercs, X_wercs))
        train = pd.DataFrame(trainWC)

      return train

    def install_rpackages(self):
      string = """

      U1 <- function(){

          install.packages("devtools")
          library(devtools)

          install.packages(c("operators", "class", "fields", "ROCR", "Hmisc", "performanceEstimation"))

          install.packages(c("zoo","xts","quantmod"))

          install.packages( "https://cran.r-project.org/src/contrib/Archive/DMwR/DMwR_0.4.1.tar.gz", repos=NULL, type="source" )

          install.packages("IRon")
          install_github("rpribeiro/uba")

          library(IRon)
          library(uba)

      }

      """
      powerpack = SignatureTranslatedAnonymousPackage(string, "powerpack")

      powerpack.U1()


    def scores(self, y, y_test, y_pred):

      uba = importr("uba")
      iron = importr("IRon")

      ph = uba.phi_control(y)
      ls = uba.loss_control(y)
      sera = iron.sera(y_test, y_pred, phi_trues = uba.phi(y_test,ph))
      F1 = uba.util(y_pred, y_test, ph, ls, uba.util_control(umetric="Fm", beta=1, event_thr=0.8))

      scores_ = list([F1, sera])
      return pd.DataFrame(scores_,
                columns = [''],
                index = ['fscore', 'sera'])

    def repeatedKfold(self, X, y, dataset, n_splits=10, n_repeats=2, random_state=42, pipeline=None, param_grid=None) :
      rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
      all_result = []

      strategys = {"SG":{"samp_method":["balance", "extreme"]},
                  "RU":{"C.perc":["balance", "extreme"]},
                  #  "RO":{"C.perc":["balance", "extreme"]},
                  #  "SMT":{"C.perc":["balance", "extreme"]},
                  #  "GN":{"C.perc":["balance", "extreme"], "pert":[0.05, 0.1, 0.5]},
                  #  "WC":{"over":[0.5, 0.8], "under":[0.5, 0.8]},
                  #  'None': {None}
                  }

      for strategy in strategys:

          data_frame = []
          params = strategys[strategy]

          keys = sorted(params)

          if strategy != "None":
            combinations = it.product(*(params[Name] for Name in keys))
          else:
            combinations = ['None']
          for c in list(combinations):
            score_perc = []
            for train_index, test_index in rkf.split(X, y):

              X_train, X_test = X[train_index], X[test_index]
              y_train, y_test = y[train_index], y[test_index]

              train = np.column_stack((y_train, X_train))
              train = pd.DataFrame(train)
              # pd.DataFrame(train).to_csv("train.csv", index=False)
              # train = pd.read_csv("train.csv")

              if c != 'None':
                try:
                  train = self.balance(train, strategy, c)
                except ValueError:
                  pass

              X_train = train.drop([train.columns[0]], axis = 1)
              y_train  = train[train.columns[0]]

              X_train = X_train.to_numpy()
              y_train = y_train.to_numpy()

              grid_search = GridSearchCV(pipeline, cv=rkf, param_grid=param_grid)
              grid_search.fit(X_train, y_train)
              y_pred  = grid_search.predict(X_test)

              path = dataset
              head, tail = os.path.split(path)

              test = np.column_stack((test_index, y_test))
              pred = np.column_stack((test_index, y_pred))


              score_perc.append(self.scores(y, y_test, y_pred).T)

            df = pd.concat(score_perc)

            values = [tail,
                      str(df.fscore.mean().round(3))+ "({})".format(df.fscore.std().round(3)),
                      str(df.sera.mean().round(3))+ "({})".format(df.sera.std().round(3))]

            scores_df = pd.DataFrame([values], columns=["dataset", "fscore", "sera"])

            if len(keys) > 1:
              scores_df[keys[0]]=c[0]
              scores_df[keys[1]]=c[1]
              scores_df['strategy']=strategy
            else:
              scores_df[keys[0]]=c[0]
              scores_df['strategy']=strategy

            data_frame.append(scores_df)
          data_frame = pd.concat(data_frame)

          # data_frame.to_csv('result_{}_{}.csv'.format(strategy, str(pipeline.steps[0][1]).split('(')[0]), index = False)
          all_result.append(data_frame)
      all_result = pd.concat(all_result).reset_index(drop=True)
      all_result['clf'] = str(pipeline.steps[0][1]).split('(')[0]

      return all_result

    def pipe_generation(self):
      clf_param = dict()
      for clf in [BaggingRegressor(DecisionTreeRegressor()), DecisionTreeRegressor(),
                  # MLPRegressor(max_iter=200), RandomForestRegressor(), SVR(), XGBRegressor(verbosity=0)
                  ]:
          clf_param[str(clf).split('(')[0]] = clf

      pipes_params = []

      # for clf,  param_grid in zip([BaggingRegressor(DecisionTreeRegressor()), DecisionTreeRegressor(), MLPRegressor(max_iter=200), RandomForestRegressor(), SVR(), XGBRegressor(verbosity=0)],


      #                 [{'clf__base_estimator__min_samples_split': [20], 'clf__max_samples':[0.5]},
      #                  {'clf__min_samples_split': [20]},
      #                  {'clf__learning_rate_init': [0.1],'clf__momentum': (0.2, 0.7),'clf__tol': (0.01, 0.05)},
      #                  {'clf__n_estimators': [550, 1500], 'clf__max_features': [5]},
      #                  {'clf__gamma': [0.01, 0.001], 'clf__C': [10, 300]},
      #                  {'clf__eta': [0.01], 'clf__max_depth': (10, 15), 'clf__colsample_bytree': (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), 'clf__num_round': [25]}]):

      for clf,  param_grid in zip([BaggingRegressor(DecisionTreeRegressor()), DecisionTreeRegressor()],


                      [{'clf__base_estimator__min_samples_split': [20], 'clf__max_samples':[0.5]},
                      {'clf__min_samples_split': [20]},]):

        configs = []
        clf = str(clf).split('(')[0]
        for p in param_grid:
            aux = p
            for i in param_grid[p]:
              aux += '+'+str(i)
            clf += '|'+aux
        configs.append(clf)

        for config in configs:

          pipeline = Pipeline([('clf', clf_param[config.split('|')[0]])])
          params = config.split('|')

          param_grid = {}
          t, t1 = len(params), 0
          for p in range(len(params)):
            values = ()
            if len(params[p].split('+')) > 2:
              a = params[p].split('+')[1:]
              for j in a:
                if '0.' in j:
                  values += (float(j),)
                else:
                  values += (int(j),)

              param_grid[params[p].split('+')[0]] = values

            else:

              if t1 == t:
                if '0.' in params[p].split('+')[1]:
                  param_grid[params[p].split('+')[0]] = [params[p].split('+')[1]]
                else:
                  param_grid[params[p].split('+')[0]] = [params[p].split('+')[1]]
              elif t1 < t:
                for l in params[t1].split('+')[1:]:

                  if '0.' in l:
                    param_grid[params[t1].split('+')[0]] = [float(l)]
                  else:
                    param_grid[params[t1].split('+')[0]] = [int(l)]

            t1 += 1

        pipes_params.append([pipeline, param_grid])
      return pipes_params

    def select_best(self, df):
      best_result = []
      for metric in ['fscore', 'sera']:
        df['value'] = df[metric].str.extract(r'^([\d.]+)\(')
        df['value'] = df['value'].astype(float)
        if metric == 'fscore':
        # Encontrar o maior valor
          best_value = df['value'].max()
        else:
          best_value = df['value'].min()
        best_value = df.loc[df['value'] == best_value, ['dataset', 'clf', 'strategy', metric]].reset_index(drop=True)[:1]
        best_value.columns = ['dataset', 'clf', 'strategy', 'score']
        best_value.insert(3, 'metric', metric)
        best_result.append(best_value)
      best_result = pd.concat(best_result)
      return best_result

    def fit(self, x_train, y_train):
        """
        Fit the meta-model.

        Parameters
        ----------
        x_train : pandas.DataFrame
            The training data.
        y_train : pandas.Series
            The training labels.

        Returns
        -------
        sklearn.ensemble.RandomForestClassifier
            The trained meta-model.
        """
        self.meta_model = RandomForestClassifier()
        self.meta_model.fit(x_train, y_train)

    def predict(self, x_test):
        """
        Make predictions using the meta-model.

        Parameters
        ----------
        model : sklearn.ensemble.RandomForestClassifier
            The meta-model.
        x_test : pandas.DataFrame
            The test data.

        Returns
        -------
        numpy.ndarray
            The predicted labels.
        """
        pred = self.meta_model.predict(x_test)[0]
        return pred

    def generation(self, X, Y):
        """
        Generate predictions using leave-one-out cross-validation.

        Parameters
        ----------
        X : pandas.DataFrame
            The dataset.
        Y : pandas.Series
            The labels.

        Returns
        -------
        numpy.ndarray
            The predictions.
        """

        loo = LeaveOneOut()
        loo.get_n_splits(X)

        y_pred = []
        for i, (train_index, test_index) in enumerate(loo.split(X, Y)):
            # Split the data into training and test sets.
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

            # Fit the meta-model on the training data.
            self.fit(x_train, y_train)

            # Make predictions on the test data.
            y_pred.append(self.predict(x_test))

        return np.array(y_pred)

    def evaluation(self, y_true, y_pred):
        """
        Evaluate the performance of the model.

        Parameters
        ----------
        y_true : pandas.Series
            The ground truth labels.
        y_pred : pandas.Series
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The evaluation scores.
        """
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')

        values = [acc, f1, precision, recall]
        return values

    def independent_training(self, m):

        """
        Train and evaluate the model independently for each label.

        Returns
        -------
        pd.DataFrame
            The evaluation scores for each label.
        """

        # Select the numeric features.
        X = m.select_dtypes(exclude=['object'])

        # Create a dictionary to map each label to its column name.
        y_labels = {'y_r': 'strategy.SERA', 'y_l': 'model.SERA'}

        # Iterate over each label.
        independent_evaluations = []
        for y_label in y_labels:
            # Get the ground truth labels for the current label.
            y_true = m[[y_labels[y_label]]]

            # Generate predictions for the current label.
            y_pred = self.generation(X, y_true)

            # Evaluate the predictions for the current label.
            independent_values = self.evaluation(y_true, y_pred)

            # Create a DataFrame for the evaluation scores of the current label.
            independent_values = pd.DataFrame([independent_values],
                                              columns=['acc', 'f1', 'precision', 'recall'])

            # Add the label name to the DataFrame.
            independent_values.insert(0, 'label', y_label)

            # Append the DataFrame to the list of independent evaluations.
            independent_evaluations.append(independent_values)

        # Concat the list of independent evaluations into a DataFrame.
        independent_evaluations = pd.concat(independent_evaluations)

        # Add a column with approach name
        independent_evaluations.insert(0, 'approach', 'independent')

        return independent_evaluations

    def model_first(self, m):
      """
      Train the model first, then the strategy.

      Returns
      -------
      pd.DataFrame
          The evaluation scores for each label.
      """

      # Select the features that are not object dtype.
      X = m.select_dtypes(exclude=['object'])

      # Create a dictionary that maps the label name to the column name.
      y_labels = {'y_r': 'strategy.SERA', 'y_l': 'model.SERA'}

      # Get the ground truth labels for the first label.
      y_true = m[[y_labels['y_l']]]

      # Generate predictions for the first label using leave-one-out cross-validation.
      y_pred = self.generation(X, y_true)

      model_first_evaluations = []

      # Evaluate the predictions for the first label.
      model_first_values = self.evaluation(y_true, y_pred)

      # Create a DataFrame to store the evaluation scores for the first label.
      model_first_values = pd.DataFrame([model_first_values],
                                        columns=['acc', 'f1', 'precision', 'recall'])
      model_first_values.insert(0, 'label', 'y_l')

      model_first_evaluations.append(model_first_values)

      # Add the predicted labels for the first label to the dataset.
      X['y_l'] = y_pred

      # Get the ground truth labels for the second label.
      y_true = m[[y_labels['y_r']]]

      # One-hot encode the predicted labels for the first label.
      column_to_encoder = 'y_l'
      encoder = OneHotEncoder(sparse=False)
      encoded_column = encoder.fit_transform(X[[column_to_encoder]])
      columns_one_hot = encoder.get_feature_names_out([column_to_encoder])
      encoded_df = pd.DataFrame(encoded_column, columns=columns_one_hot)

      # Concatenate the one-hot encoded predictions with the original dataset.
      X = pd.concat([X.drop(column_to_encoder, axis=1), encoded_df], axis=1)

      # Generate predictions for the second label using the updated dataset.
      y_pred = self.generation(X, y_true)

      # Evaluate the predictions for the first label.
      model_first_values = self.evaluation(y_true, y_pred)

      # Evaluate the predictions for the second label.
      model_first_values = pd.DataFrame([model_first_values],
                                        columns=['acc', 'f1', 'precision', 'recall'])
      model_first_values.insert(0, 'label', 'y_r')

      model_first_evaluations.append(model_first_values)

      # Concat the list of model_first evaluations into a DataFrame.
      model_first_evaluations = pd.concat(model_first_evaluations)

      # Add a column with approach name
      model_first_evaluations.insert(0, 'approach', 'model_first')

      # Return the evaluation scores for both labels.
      return model_first_evaluations

    def strategy_first(self, m):
      """
      Train the model first, then the strategy.

      Returns
      -------
      pd.DataFrame
          The evaluation scores for each label.
      """

      # Select the features that are not object dtype.
      X = m.select_dtypes(exclude=['object'])

      # Create a dictionary that maps the label name to the column name.
      y_labels = {'y_r': 'strategy.SERA', 'y_l': 'model.SERA'}

      # Get the ground truth labels for the first label.
      y_true = m[[y_labels['y_r']]]

      # Generate predictions for the first label using leave-one-out cross-validation.
      y_pred = self.generation(X, y_true)

      strategy_evaluations = []

      # Evaluate the predictions for the first label.
      strategy_values = self.evaluation(y_true, y_pred)

      # Create a DataFrame to store the evaluation scores for the first label.
      strategy_values = pd.DataFrame([strategy_values],
                                     columns=['acc', 'f1', 'precision', 'recall'])
      strategy_values.insert(0, 'label', 'y_r')

      strategy_evaluations.append(strategy_values)

      # Add the predicted labels for the first label to the dataset.
      X['y_r'] = y_pred

      # Get the ground truth labels for the second label.
      y_true = m[[y_labels['y_l']]]

      # One-hot encode the predicted labels for the first label.
      column_to_encoder = 'y_r'
      encoder = OneHotEncoder(sparse=False)
      encoded_column = encoder.fit_transform(X[[column_to_encoder]])
      columns_one_hot = encoder.get_feature_names_out([column_to_encoder])
      encoded_df = pd.DataFrame(encoded_column, columns=columns_one_hot)

      # Concatenate the one-hot encoded predictions with the original dataset.
      X = pd.concat([X.drop(column_to_encoder, axis=1), encoded_df], axis=1)

      # Generate predictions for the second label using the updated dataset.
      y_pred = self.generation(X, y_true)

      # Evaluate the predictions for the first label.
      strategy_values = self.evaluation(y_true, y_pred)

      # Evaluate the predictions for the second label.
      strategy_values = pd.DataFrame([strategy_values],
                                     columns=['acc', 'f1', 'precision', 'recall'])
      strategy_values.insert(0, 'label', 'y_l')

      strategy_evaluations.append(strategy_values)

      # Concat the list of model_first evaluations into a DataFrame.
      strategy_evaluations = pd.concat(strategy_evaluations)

      # Add a column with approach name
      strategy_evaluations.insert(0, 'approach', 'strategy_first')

      # Return the evaluation scores for both labels.
      return strategy_evaluations
