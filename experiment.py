""" Experiment fits models to one dataset
"""
import os
import logging
import time
from pdb import set_trace
import pickle

import pandas as pd
import pyro

from utils import basename_sans_ext

class BaseExperiment(object):

    def __init__(self, models, model_names, dataset,
                 req_freq = 1, max_iter = 1000, record_param_freq = None):
        assert(len(models) == len(model_names))
        self.model_dict = dict(zip(model_names, models))
        self.dataset = dataset
        self.req_freq = req_freq
        self.max_iter = max_iter
        self.record_param_freq = record_param_freq

    def train_one_step(self, model, data_dict, t):
        res = model.train_one_step(data_dict)
        res['timestamp'] = time.time()
        if t % self.req_freq == 0:
            logging.info('Training result at {t} is {res}.'.format(t = t, res = res))
        return(res)

    def train_and_save(self, export_dir):
        self.record_dict = {}
        training_data = self.dataset.get_training_data()

        for model_name in self.model_dict:
            pyro.clear_param_store()
            self.record_dict[model_name] = []
            logging.info('Training model {}.'.format(model_name))

            for t in range(self.max_iter):
                self.record_dict[model_name].append(
                    self.train_one_step(self.model_dict[model_name], training_data, t)
                )
                # if self.record_param_freq is set, save params
                # per record_param_freq inteval
                if self.record_param_freq and (t % self.record_param_freq == 0):
                    self.save_params(model_name, param_dir, t)

            self.save_params(model_name, export_dir, t)
            self.save_record(self.record_dict[model_name], model_name, export_dir, t)

    def save_params(self, model_name, export_dir, t):
        param_dir = os.path.join(export_dir, 'params')
        if not os.path.isdir(param_dir):
            logging.info('Creating directory {}.'.format(param_dir))
            os.makedirs(param_dir)

        if t == 'final':
            out_path = os.path.join(param_dir, '{model_name}_final.params'.format(model_name = model_name))
        else:
            out_path = os.path.join(param_dir, '{model_name}_it{t:04d}.params'.format(model_name = model_name, t = t))

        param_store = pyro.get_param_store()
        param_store.save(out_path)
        logging.info('Wrote to {}.'.format(out_path))

        return out_path

    def save_record(self, record, model_name, export_dir, t):
        out_path = os.path.join(export_dir, '{model_name}_it{t:04d}_record.pkl'.format(model_name = model_name, t = t))
        pickle.dump(
            pd.DataFrame(record),
            open(out_path, 'wb')
        )
        logging.info('Wrote to {}.'.format(out_path))
        return out_path

class AutoStopExperiment(BaseExperiment):

    def train_and_save(self, export_dir, auto_stop = False, stop_threshold = 0.001):
        self.record_dict = {}
        training_data = self.dataset.get_training_data()

        for model_name in self.model_dict:
            pyro.clear_param_store()
            self.record_dict[model_name] = []
            logging.info('Training model {}.'.format(model_name))

            for t in range(self.max_iter):
                self.record_dict[model_name].append(
                    self.train_one_step(self.model_dict[model_name], training_data, t)
                )
                # if self.record_param_freq is set, save params
                # per record_param_freq inteval
                if self.record_param_freq and (t % self.record_param_freq == 0):
                    self.save_params(model_name, export_dir, t)

                if auto_stop and self.at_elbow_point(self.record_dict[model_name], stop_threshold = stop_threshold):
                    logging.info('Training auto stopped at t = {}.'.format(t))
                    break

            self.save_params(model_name, export_dir, t)
            self.save_params(model_name, export_dir, t = 'final') # save a second copy for naming consistency
            self.save_record(self.record_dict[model_name], model_name, export_dir, t)

    def at_elbow_point(self, record, n_sample = 5, stop_threshold = 0.001):
        assert(n_sample >= 2)
        # record not long enough
        if (len(record) < n_sample): return False
        # if the average of the last
        # n_sample - 1 drops are less than stop_threshold
        # of the first drop, then return True
        first_drop = record[0]['elbo'] - record[1]['elbo']
        average_drop = (record[-1 - n_sample + 1]['elbo'] - record[-1]['elbo']) / (n_sample - 1)
        return average_drop < stop_threshold * first_drop

class EvalExperiment(AutoStopExperiment):

    def __init__(self, models, model_names, dataset,
                 req_freq = 1, max_iter = 1000, record_param_freq = None):
        super().__init__(models, model_names, dataset, req_freq, max_iter, record_param_freq)
        self.param_paths = {}
        self.record_paths = {}

    def assemble_prediction_df(self, model, data_dict):
        prob = model.predict(data_dict).detach().numpy()
        df = pd.DataFrame({
                'ques_id' : data_dict['ques_id'].detach().numpy(),
                'stu_id' : data_dict['stu_id'].detach().numpy(),
                'correct' : data_dict['correct'].detach().numpy(),
                'prob' : prob,
            })
        return df

    def make_prediction(self, export_dir):
        for model_name in self.model_dict:
            if model_name not in self.param_paths:
                logging.info('No path to parameters for {} found.'.format(model_name))
                continue
            self.load_param_and_predict(model_name, self.param_paths[model_name], export_dir)

    def load_param_and_predict(self, model_name, param_path, export_dir):
        pyro.clear_param_store()
        param_store = pyro.get_param_store()
        param_store.load(param_path)
        logging.info('Loaded parameters for {} from {}.'.format(model_name, param_path))

        model = self.model_dict[model_name]
        prediction = {}
        prediction['training'] = self.assemble_prediction_df(model, self.dataset.get_training_data())
        if self.dataset.get_use_tvt():
            prediction['validation'] = self.assemble_prediction_df(model, self.dataset.get_validation_data())
            prediction['test'] = self.assemble_prediction_df(model, self.dataset.get_test_data())

        if self.dataset.stu_encoder:
            for data_name in prediction:
                prediction[data_name]['stu_id_original'] = self.dataset.stu_encoder.inverse_transform(prediction[data_name]['stu_id'])
        if self.dataset.ques_encoder:
            for data_name in prediction:
                prediction[data_name]['ques_id_original'] = self.dataset.ques_encoder.inverse_transform(prediction[data_name]['ques_id'])

        out_path = os.path.join(export_dir, '{prefix}_prediction.pkl'.format(prefix = basename_sans_ext(param_path)))
        with open(out_path, 'wb') as f:
            pickle.dump(prediction, f)
        logging.info('Wrote to {}.'.format(out_path))

    def save_params(self, model_name, export_dir, t):
        out_path = super().save_params(model_name, export_dir, t)
        self.param_paths[model_name] = out_path

    def save_record(self, record, model_name, export_dir, t):
        out_path = super().save_record(record, model_name, export_dir, t)
        self.record_paths[model_name] = out_path
