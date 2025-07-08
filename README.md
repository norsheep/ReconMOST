# ReconMOST

## Usage

This section of the README walks through how to train and sample from a model.

### Installation

Clone this repository and navigate to it in your terminal. Then run:

```
pip install -e .
```

This should install the `improved_diffusion` python package that the scripts depend on.

### Preparing Data

We use datasets CMIP6 and EN4. For creating and using your own datasets, please refer to improved_diffusion/image_datasets_v2.py. You only need to give the data-path when training and sampling.

### Training

To train your model, you should first decide some hyperparameters. We will split up our hyperparameters into three groups: model architecture, diffusion process, and training flags. Here is a training example:

```
DATA_DIR=YOUR_DATA_PATH
MODEL_FLAGS="--in_channels 42 --image_size 180 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 4"
export DIFFUSION_BLOB_LOGDIR=YOUR_SAVE_PATH
python ./scripts/image_train_v2.py --data_dir $DATA_DIR $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

### Sampling

The above training script saves checkpoints to `.pt` files in the logging directory. These checkpoints will have names like `ema_0.9999_200000.pt` and `model200000.pt`. You will likely want to sample from the EMA models, since those produce much better samples.

Once you have a path to your model, you can generate a large batch of samples like so:

```
MODEL_PATH=YOUR_MODEL_PATH
SPARSE_DATA_PATH=YOUR_DATA_PATH
ADJUST_FLAGS="--num_samples 4 --use_sigma False --grad_scale 4 --use_softmask False --guided_rate 0.075 --loss_model l2"
export DIFFUSION_SAMPLE_LOGDIR=YOUR_SAVE_PATH
MODEL_FLAGS="--in_channels 42 --image_size 180 --image_size_H 173 --image_size_W 360 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --use_ddim False  --clip_denoised True"
SAMPLE_FLAGS="--batch_size 4"
python ./scripts/guided_sample_analysis.py --model_path $MODEL_PATH --sparse_data_path $SPARSE_DATA_PATH $MODEL_FLAGS $DIFFUSION_FLAGS  $SAMPLE_FLAGS $ADJUST_FLAGS
```

```bash
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"
```
## Reference

This work is built upon [guided-diffusion](https://github.com/openai/guided-diffusion). Thanks to their awesome work.
