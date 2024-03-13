import os, gc, argparse, subprocess, json
from tensorflow import keras
import numpy as np
import pandas
from training_data import load_training_data, MultiLevelTrainingData
import config



def below_threshold(y_true, y_pred, threshold=0.1):
    kb = keras.backend
    diff = kb.abs(y_true - y_pred)
    return kb.mean(kb.cast(kb.all(diff < threshold, axis=-1), 'float32'))
    

class TrainingHistory:
    """Stores history of train/validation loss during training"""
    def __init__(self, history_file=None):
        self.history_file = history_file
        if history_file is not None and os.path.exists(history_file):
            self.history = pandas.read_pickle(history_file)
        else:
            self.history = pandas.DataFrame({'batch': [], 'train_mse': []})

        cols = [col for col in self.history.columns if col.startswith('val_')]
        self.cols_by_metric = {
            'mse': [col for col in cols if col.endswith('0_mse')],
            'xy_mse': [col for col in cols if col.endswith('xy_mse')],
            'z_mse': [col for col in cols if col.endswith('z_mse')],
        }
        self.metrics = list(self.cols_by_metric.keys())
        self.levels = [col.split('_')[1] for col in self.cols_by_metric['mse']]

    def __len__(self):
        return len(self.history)
    
    def append(self, record):
        record = {k:[x] for k,x in record.items()}
        self.history = pandas.concat([self.history, pandas.DataFrame(record)], ignore_index=True)
        self.save()        

    def save(self):
        if self.history_file is not None:
            self.history.to_pickle(self.history_file)

    def get_slopes(self, size=5):
        slopes = {}
        for col in self.history.columns:
            if col != 'batch':
                x = np.array(self.history['batch'][-size:])
                y = np.array(self.history[col][-size:])
                if len(x) > 1:
                    slopes[col] = np.polyfit(x, y, deg=1)[0] / y.mean()
        return slopes

    def plot_loss(self, metrics=None, ax=None):
        if metrics is None:
            metrics = self.metrics
        n_levels = len(self.levels)
        n_metrics = len(metrics)
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(n_levels, n_metrics, figsize=(n_metrics*3, n_levels*3))
        for i, level in enumerate(self.levels):
            for j, metric in enumerate(metrics):
                ax[i, j].axhline(0.005, color=(0.7, 0.7, 0.7), linewidth=0.5)
                ax[i, j].plot(self.history[f'val_{level}_{metric}'])
                ax[i, j].set_yscale('log')
                # ax.legend()


class ProgressLogger(keras.callbacks.Callback):
    def __init__(self, save_path):
        self.log_file = os.path.join(save_path, 'last_batch.txt')
        keras.callbacks.Callback.__init__(self)
    
    def on_train_batch_end(self, batch, logs=None):
        open(self.log_file, 'w').write(f"{batch}\n")


class PeriodicValidation(keras.callbacks.Callback):
    def __init__(self, n_iter, history_file, training_data, validation_data, threshold=1e-5):
        self._n_iter = n_iter
        self.last_batch = None
        self._training_data = training_data
        self._validation_data = validation_data
        self._validation_data_arr = validation_data.get_arrays()
        self.history = TrainingHistory(history_file)
        self.threshold = threshold
        keras.callbacks.Callback.__init__(self)
        
    def compute_mse(self, actual_pos, predicted_pos):
        full_mse = np.mean(np.square(actual_pos - predicted_pos))
        xy_mse = np.mean(np.square(actual_pos[:, 1:] - predicted_pos[:, 1:]))
        z_mse = np.mean(np.square(actual_pos[:, 0] - predicted_pos[:, 0]))
        return {'mse': full_mse, 'xy_mse': xy_mse, 'z_mse': z_mse}
    
    def on_train_batch_end(self, batch, logs=None):
        self.last_batch = batch
        if batch % self._n_iter == 0:
            self.run_validation(batch)
            self.history.save()

    def run_validation(self, batch):
        print(f"\nBatch {batch}")
        record = {'batch': batch}

        if self._training_data.last_batch is not None:
            train_images, train_pos = self._training_data.last_batch        
            pred_train_pos = self.model.predict(train_images, verbose=0)
            train_mse = self.compute_mse(train_pos, pred_train_pos)
        else:
            train_mse = None
        for k,v in train_mse.items():
            record[f"train_{k}"] = v
                
        val_images, val_pos = self._validation_data_arr
        pred_val_pos = self.model.predict(val_images, verbose=0)
        for level_name, level_slice in self._validation_data.level_slices.items():
            val_mse = self.compute_mse(val_pos[level_slice], pred_val_pos[level_slice])
            for k,v in val_mse.items():
                record[f"val_{level_name}_{k}"] = v

        self.history.append(record)

        loss_slopes = self.history.get_slopes(size=10)
        if len(loss_slopes) > 0:
            train_slopes = [loss_slopes[f"train_{k}"] for k in ['mse', 'xy_mse', 'z_mse']]
            print(f"    Training MSE (xyz, xy, z): {train_mse['mse']} {train_mse['xy_mse']} {train_mse['z_mse']}  Slopes: {train_slopes}")
            levels = [(f'Validation {x}', 'val_'+x) for x in self._validation_data.levels.keys()]
            for level_name, level_key in levels:
                level_slopes = [loss_slopes[f"{level_key}_{k}"] for k in ['mse', 'xy_mse', 'z_mse']]
                mse_str = f"MSE (xyz, xy, z): {record[level_key+'_mse']:0.3g} {record[level_key+'_xy_mse']:0.3g} {record[level_key+'_z_mse']:0.3g}"
                slope_str = f"Slopes: {' '.join(['%0.3g'%x for x in level_slopes])}"
                print(f"    {level_name}  {mse_str}  {slope_str}")

            val_slopes = [loss_slopes[k] for k in loss_slopes.keys() if k.startswith('val')]
            if len(self.history) >= 10 and np.all(np.array(val_slopes) > -self.threshold):
                print(f"Validation loss slopes crossed threshold: {val_slopes}; terminating training early")
                self.model.stop_training = True
        
        gc.collect() 
        keras.backend.clear_session()


class PeriodicModelSave(keras.callbacks.Callback):
    def __init__(self, n_iter, filename):
        self._n_iter = n_iter
        self.filename = filename
        keras.callbacks.Callback.__init__(self)
        
    def on_train_batch_end(self, batch, logs=None):
        if batch > 0 and batch % self._n_iter == 0:
            print(f"saving weights to {self.filename}")
            self.model.save_weights(self.filename)


def fit_model(training_data, save_path, batch_size=64, rate_scheduler=None, training_level=None, model_type=None, load_model=None, load_weights=None, 
              train_depth=1.0, optimizer=None, learning_rate=None, epochs=10, allow_cpu=False, model_opts=None):
    if os.path.exists(save_path):
        raise Exception(f"Save path {save_path} already exists")

    # imports slowly due to tensorflow; wait until after argument parsing to import
    from model import PipetteDetectionModel

    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUS:", gpus, allow_cpu)
    if not allow_cpu and len(gpus) == 0:
        raise Exception("Exiting; no GPU available (use --allow-cpu to override)")

    # make sure one of model_type or load_model is specified
    if model_type is None and load_model is None:
        raise Exception("Must specify either model_type or load_model")

    if isinstance(training_data, str):
        all_data = load_training_data(training_data)
    else:
        all_data = training_data
    
    if isinstance(all_data, MultiLevelTrainingData) and training_level is None:
        raise Exception("Multiple training levels present; must specify training-level")
    training_data, validation_data = all_data.split([0.995, 0.005])
    if training_level is not None:
        training_data = training_data.get_level(training_level)
    print(f"Loaded {len(training_data)} training examples and {len(validation_data)} validation examples")

    input_shape = training_data[0][0].shape

    model_opts = model_opts if model_opts is not None else {}
    if load_model is not None:
        model = PipetteDetectionModel(load_model=load_model, model_opts=model_opts)
    else:
        model = PipetteDetectionModel(model_opts={'model_type': model_type, 'input_shape': input_shape, **model_opts})

    if load_weights is not None:
        model.load_weights(load_weights)

    model.fit(
        training_data, 
        validation_data, 
        train_depth=train_depth,
        optimizer=optimizer, 
        learning_rate=learning_rate,
        batch_size=batch_size,
        rate_scheduler=rate_scheduler,
        epochs=epochs,
        save_path=save_path,
        val_interval=100,
        save_interval=1000,
    )
    
    return model, training_data
