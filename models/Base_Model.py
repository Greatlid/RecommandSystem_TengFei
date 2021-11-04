import torch
import torch.nn as nn
from sklearn.metrics import *
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
from utils.Sampler import *
from tensorflow.python.keras.callbacks import CallbackList
from tensorflow.python.keras.callbacks import History

class BaseModel(torch.nn.Module):
    def __init__(self, device = 'cpu', seed = 1234, gpus = None):
        super(BaseModel, self).__init__()
        torch.manual_seed(seed)
        self.regularization_weight = []
        self.reg_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        self.history = History()
        self.aux_loss = torch.zeros((1,), device=device)

    def fit(self, x=None, y=None, batch_size=256, epochs=1, verbose=1, initial_epoch=0,
            validation_data=None, shuffle=True, callbacks=None):
        """

         :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
             dictionary mapping input names to Numpy arrays.
         :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
         :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
         :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
         :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
         :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
         :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
         :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
         :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
         :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

         :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
         """

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
        else:
            val_x = []
            val_y = []


        model = self.train()
        loss_func = self.loss_func
        optim = self.optim
        if self.gpus[0] > 0:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)


        train_loader = BatchSampler(
            feature = x, label = y, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(x)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(x), len(val_y), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}

            for x_train, y_train in train_loader.GetBatch():
                y_pred = model(x_train).squeeze()
                optim.zero_grad()
                loss = loss_func(y_pred, torch.Tensor(y_train).squeeze(), reduction='sum')
                reg_loss = self.get_regularization_loss()

                total_loss = loss + reg_loss + self.aux_loss

                loss_epoch += loss.item()
                total_loss_epoch += total_loss.item()
                total_loss.backward()
                optim.step()

                if verbose > 0:
                    for name, metric_fun in self.metrics.items():
                        if name not in train_result:
                            train_result[name] = []
                        y_target, y_pred = torch.Tensor(y_train).cpu().data.numpy().astype("float64"), \
                                           y_pred.cpu().data.numpy().astype("float64")
                        if not y_pred.shape:
                            y_pred = np.array([y_pred])
                        train_result[name].append(metric_fun(y_target,
                            y_pred))




            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history


    def evaluate(self, x, y, batch_size=256):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, batch_size=256):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()
        test_loader = BatchSampler(
            feature=x, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for x_test, _ in test_loader.GetBatch():
                y_pred = model(x_test).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(np.concatenate(pred_ans)).astype("float64")

    def compile(self, optimizer = None,
                lr = 0.001,
                loss=None,
                metrics=None,
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer, lr)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_initializer(self, init_method):
        if init_method == 'xavier':
            for m in self.modules():
                if hasattr(m, 'weight'):
                    nn.init.xavier_uniform_(m.weight)
        elif init_method == 'constant':
            for m in self.modules():
                if hasattr(m, 'weight'):
                    torch.nn.init.constant_(m.weight, 0)
        elif init_method == 'uniform':
            for m in self.modules():
                if hasattr(m, 'weight'):
                    torch.nn.init.uniform_(m.weight, a=0, b=1)
        elif init_method == 'normal':
            for m in self.modules():
                if hasattr(m, 'weight'):
                    torch.nn.init.normal_(m.weight, mean=0, std=1)
        elif init_method == 'dirac':
            for m in self.modules():
                if hasattr(m, 'weight'):
                    torch.nn.init.dirac_(m.weight, 0)
        else:
            return


    def _get_optim(self, optimizer, lr):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=lr)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters(), lr=lr)  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters(), lr=lr)  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters(), lr=lr)
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                self.metrics_names.append(metric)
        return metrics_

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha