import os, json
from tensorflow import keras
from training import PeriodicValidation, PeriodicModelSave


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

    def fit(self, training_data, validation_data, train_depth, optimizer=None, 
            learning_rate=None, batch_size=64, rate_scheduler=None, epochs=1, 
            save_path=None, val_interval=100, save_interval=1000):
        """Fit the model to *training_data* and validate on *validation_data*.

        Parameters
        ----------
        training_data : TrainingData
            Training data to fit to
        validation_data : TrainingData
            Validation data to validate on
        train_depth : float
            Fraction of the model layers after which to begin training (0.0 = all layers, 1.0 = no layers)
        optimizer : keras.optimizers.Optimizer | None
            Optimizer to use for training (default: Adam)
        learning_rate : float
            Learning rate to use for training
        batch_size : int
            Batch size to use for training
        rate_scheduler : dict | None
            Options to generate a learning rate scheduler {'decay': 0.9, 'delay': 2}
        epochs : int
            Number of epochs to train for
        save_path : str
            Path to save the model to
        val_interval : int
            Number of batches between validation checks
        save_interval : int
            Number of batches between model saves
        """
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

        if optimizer is None:
            optimizer_args = {}
            if learning_rate is not None:
                optimizer_args['learning_rate'] = learning_rate
            otimizer = keras.optimizers.Adam(**optimizer_args)
        self.model.compile(
            optimizer=optimizer, 
            loss='mse',
        )

        self.validator = PeriodicValidation(
            n_iter=val_interval, 
            history_file=os.path.join(save_path, 'training_history.json'),
            training_data=training_data,
            validation_data=validation_data,
        )
        callbacks = [self.validator]

        if rate_scheduler is not None:
            def scheduler(epoch, learning_rate):
                if epoch < rate_scheduler['delay']:
                    return learning_rate
                else:
                  return learning_rate * rate_scheduler['decay']
            callbacks.append(keras.callbacks.LearningRateScheduler(scheduler))

        weights_path = os.path.join(save_path, 'fit_weights')
        if save_path is not None:
            callbacks.append(PeriodicModelSave(save_interval, weights_path))

        try:
            self.model.fit(    # convert _ to - for kwds
                training_data.generator(batch_size=batch_size), 
                steps_per_epoch=len(training_data)//batch_size, 
                epochs=epochs, 
                batch_size=batch_size,
                callbacks=callbacks,
            )
        finally:
            if save_path is not None:
                full_save_path = os.path.join(save_path, 'fit_model')
                print(f"Saving final model to {full_save_path}..")
                self.model.save_weights(weights_path)
                self.model.save(full_save_path)
