import argparse, subprocess, sys
import config


def get_tf_container_id():
    container_id = subprocess.check_output(["docker", "ps", f"--filter=ancestor={config.docker_image}", "-q"]).decode("utf-8").strip()
    if len(container_id) == 0:
        raise Exception(f"Could not find running container with image {config.docker_image}")
    return container_id


def train_model_in_subprocess(args):
    if config.use_docker:
        cmd = ["docker", "exec", "-it", "pipette-detection", "python", "/tf/pipette-detection/train_model.py"] + args
    else:
        cmd = ["python", __file__] + args
    print(" ".join(cmd))
    proc = subprocess.Popen(cmd)
    return proc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-data', type=str, help="path to training data")
    parser.add_argument('--training-level', type=str, help="training difficulty level, if multiple levels are present")
    parser.add_argument('--save-path', type=str, default=None, help="path to save model, weights, history, and configuration")
    parser.add_argument('--load-model', type=str, default=None, help="path to load model")
    parser.add_argument('--load-weights', type=str, default=None, help="path to load weights")
    parser.add_argument('--model-type', type=str, default='ResNet101V2', help="Base model type (ResNet101V2, ResNet50, VGG16, VGG19, ...)")
    parser.add_argument('--learning-rate', type=float, default=None, help='learning rate (0.001 is default)')
    parser.add_argument('--train-depth', type=float, default=1.0, help='base model depth at which to begin training (0.0 means train all; 1.0 means train none)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--allow-cpu', default=False, action='store_true', help='Allow training without GPU (otherwise, exit if no GPU is available)')
    parser.add_argument('--use-docker', default=False, action='store_true', help='Run training in docker container')
    args = parser.parse_args()

    if args.use_docker:
        args = [arg for arg in sys.argv[1:] if arg != '--use-docker']
        proc = train_model_in_subprocess(args)
        proc.wait()
    else:
        from training import fit_model
        pass_args = {k:v for k,v in args.__dict__.items() if k not in ['use_docker'] and v is not None}
        fit_model(**pass_args)
