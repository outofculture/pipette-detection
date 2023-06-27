import os, argparse
from training_data import TrainingData, Normalizer


parser = argparse.ArgumentParser()
parser.add_argument('--training-data', type=str, help="path to training data")
parser.add_argument('--save-path', type=str, default=None, help="path to save model, weights, history, and configuration")
parser.add_argument('--load-model', type=str, default=None, help="path to load model")
parser.add_argument('--load-weights', type=str, default=None, help="path to load weights")
parser.add_argument('--model-type', type=str, default='ResNet101V2', help="Base model type (ResNet101V2, ResNet50, VGG16, VGG19, ...)")
parser.add_argument('--learning-rate', type=float, default=None, help='learning rate (0.001 is default)')
parser.add_argument('--train-depth', type=float, default=1.0, help='base model depth at which to begin training (0.0 means train all; 1.0 means train none)')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--allow-cpu', type=bool, default=False, help='Allow training without GPU (otherwise, exit if no GPU is available)')
args = parser.parse_args()

if os.path.exists(args.save_path):
    raise Exception(f"Save path {args.save_path} already exists")

# imports slowly due to tensorflow; wait until after argument parsing to import
from model import PipetteDetectionModel

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if not args.allow_cpu and len(gpus) == 0:
    raise Exception("Exiting; no GPU available (use --allow-cpu to override)")


norm = Normalizer(range=[[-100, 0, 0], [100, 500, 500]])
all_data = TrainingData(args.training_data, output_norm=norm.normalize)
training_data, validation_data = all_data.split([0.995, 0.005])
print(f"Loaded {len(training_data)} training examples and {len(validation_data)} validation examples")

batch_size = args.batch_size
input_shape = training_data[0][0].shape


if args.load_model is not None:
    model = PipetteDetectionModel(load_model=args.load_model)
else:
    model = PipetteDetectionModel(model_opts={'model_type': args.model_type, 'input_shape': input_shape})


model.fit(
    training_data, 
    validation_data, 
    train_depth=args.train_depth, 
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    epochs=args.epochs,
    save_path=args.save_path,
    val_interval=100,
    save_interval=1000,
)
