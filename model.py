import os, gc, json
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def below_threshold(y_true, y_pred, threshold=0.1):
    kb = keras.backend
    diff = kb.abs(y_true - y_pred)
    return kb.mean(kb.cast(kb.all(diff < threshold, axis=-1), 'float32'))
    

class TrainingHistory:
    """Stores history of train/validation loss during training"""
    def __init__(self, history_file=None):
        self.history_file = history_file
        self.history = {'batch': [], 'train_mse': [], 'val_mse': []}
        if history_file is not None and os.path.exists(history_file):
            with open(history_file, 'r') as fh:
                self.history = json.load(fh)

    @property
    def batch(self):
        return self.history['batch']
    
    @property
    def train_mse(self):
        return self.history['train_mse']
    
    @property
    def val_mse(self):
        return self.history['val_mse']
    
    def append(self, batch, train_mse, val_mse):
        self.history['batch'].append(batch)
        self.history['train_mse'].append(train_mse)
        self.history['val_mse'].append(val_mse)
        self.save()        

    def save(self):
        if self.history_file is not None:
            with open(self.history_file, 'w') as fh:
                hist = {k:(v.tolist() if isinstance(v, np.ndarray) else v) for k,v in self.history.items()}
                json.dump(hist, fh)

    def get_slopes(self, key, size=5):
        x = np.array(self.history['batch'][-size:])
        y = np.vstack(self.history[key][-size:])
        if len(x) < 2:
            return None
        return [np.polyfit(x, y[:,i], deg=1)[0] for i in range(y.shape[1])]


class PeriodicValidation(keras.callbacks.Callback):
    def __init__(self, n_iter, history_file, training_data, validation_data, threshold=1e-5, smoothing=0.9):
        self._n_iter = n_iter
        self._training_data = training_data
        self._validation_data = validation_data[:]
        assert self._validation_data is not None
        self.history = TrainingHistory(history_file)
        self.threshold = threshold
        self.smoothing = smoothing
        keras.callbacks.Callback.__init__(self)
        
    def compute_mse(self, actual_pos, predicted_pos):
        full_mse = np.mean(np.square(actual_pos - predicted_pos))
        xy_mse = np.mean(np.square(actual_pos[:, 1:] - predicted_pos[:, 1:]))
        z_mse = np.mean(np.square(actual_pos[:, 0] - predicted_pos[:, 0]))
        return [full_mse, xy_mse, z_mse]
    
    def on_train_batch_end(self, batch, logs=None):
        if batch % self._n_iter == 0:
            self.run_validation(batch)
            self.history.save()

    def run_validation(self, batch):
        print(f"\nBatch {batch}")

        if self._training_data.last_batch is not None:
            train_images, train_pos = self._training_data.last_batch        
            pred_train_pos = self.model.predict(train_images, verbose=0)
            train_mse = self.compute_mse(train_pos, pred_train_pos)
        else:
            train_mse = None
        
        val_images, val_pos = self._validation_data
        pred_val_pos = self.model.predict(val_images, verbose=0)
        val_mse = self.compute_mse(val_pos, pred_val_pos)

        self.history.append(batch, train_mse, val_mse)

        train_slopes = self.history.get_slopes('train_mse')
        print(f"    Training MSE (xyz, xy, z): {train_mse}  Slopes: {train_slopes}")
        val_slopes = self.history.get_slopes('val_mse')
        print(f"  Validation MSE (xyz, xy, z): {val_mse}  Slopes: {val_slopes}")

        if len(self.history.batch) >= 5 and np.all(np.array(val_slopes) > -self.threshold):
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
        if batch % self._n_iter == 0:
            print(f"saving weigts to {self.filename}")
            self.model.save_weights(self.filename)


class PipetteDetectionModel:
    def __init__(self, model_opts=None, load_model=None):
        self.model_opts = {'initial_model_path': load_model} if model_opts is None else model_opts
        
        if load_model is not None:
            self.model = keras.models.load_model(os.path.join(load_model, 'fit_model'))
        else:
            self.model = self.create_model(**({} if model_opts is None else model_opts))
        print(self.model.summary())

    @staticmethod
    def create_model(model_type, input_shape, pooling_layer=True, flatten_layer=False, dense_layers=None):
        base_model = getattr(keras.applications, model_type)(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False

        layers = [
            keras.Input(shape=input_shape),
            base_model,
        ]
        if pooling_layer:
            layers.append(keras.layers.GlobalAveragePooling2D())
        if flatten_layer:
            layers.append(keras.layers.Flatten())
        if dense_layers is not None:
            for size in dense_layers:
                layers.append(keras.layers.Dense(size, activation='relu'))

        layers.append(keras.layers.Dense(3))
        model = keras.models.Sequential(layers)

        return model

    def load_weights(self, weight_file):
        self.model.load_weights(weight_file)

    def save_weights(self, weight_file):
        self.model.save_weights(weight_file)

    def save_model(self, model_file):
        self.model.save_model(model_file)

    def fit(self, training_data, validation_data, train_depth, learning_rate=None, batch_size=64, epochs=1, save_path=None, val_interval=100, save_interval=1000):
        if save_path is not None:
            assert not os.path.exists(save_path), f"Save path {save_path} already exists"
            os.makedirs(save_path)

        fit_opts = {
            'train_depth': train_depth, 
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
        }

        with open(os.path.join(save_path, 'model_options.json'), 'w') as fh:
            json.dump(self.model_opts, fh)
        with open(os.path.join(save_path, 'fit_options.json'), 'w') as fh:
            json.dump(fit_opts, fh)
        
        base_model = self.model.layers[0]
        for i,layer in enumerate(base_model.layers):
            layer.trainable = i > len(base_model.layers) * train_depth

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
            loss='mse',
        )

        self.validator = PeriodicValidation(
            n_iter=val_interval, 
            history_file=os.path.join(save_path, 'training_history.json'),
            training_data=training_data,
            validation_data=validation_data,
        )
        callbacks = [self.validator]
        weights_path = os.path.join(save_path, 'fit_weights')
        if save_path is not None:
            callbacks.append(PeriodicModelSave(save_interval, weights_path))

        try:
            self.model.fit(
                training_data.generator(batch_size=batch_size), 
                steps_per_epoch=len(training_data)//batch_size, 
                epochs=epochs, 
                batch_size=batch_size,
                callbacks=callbacks,
            )
        finally:
            if save_path is not None:
                print("Saving final model..")
                self.model.save_weights(weights_path)
                self.model.save(os.path.join(save_path, 'fit_model'))
