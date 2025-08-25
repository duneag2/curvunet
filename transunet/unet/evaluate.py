"""
Evaluation script used to calculate accuracy of trained model
"""
import os
import h5py
import hydra
from omegaconf import DictConfig
import tensorflow as tf

import data_generator
from utils.general_utils import join_paths, set_gpus, get_gpus_count
from models.model import prepare_model
from losses.loss import dice_coef, iou
from losses.unet_loss import *


# +
def evaluate(cfg: DictConfig):
    """
    Evaluate or calculate accuracy of given model
    """
    
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    if cfg.USE_MULTI_GPUS.VALUE:
        # change number of visible gpus for evaluation
        set_gpus(cfg.USE_MULTI_GPUS.GPU_IDS)
        # change batch size according to available gpus
        cfg.HYPER_PARAMETERS.BATCH_SIZE = \
            cfg.HYPER_PARAMETERS.BATCH_SIZE * get_gpus_count

    # data generator
    val_generator = data_generator.DataGenerator(cfg, mode="VAL")

    # load training settings
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg.HYPER_PARAMETERS.LEARNING_RATE
    )
    # create model
    if cfg.USE_MULTI_GPUS.VALUE:
        # multi gpu training using tensorflow mirrored strategy
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
        )
        print('Number of visible gpu devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = prepare_model(cfg)
    else:
        model = prepare_model(cfg)

    model.compile(
        optimizer=optimizer,
        loss=unet3p_hybrid_loss,
        metrics=[dice_coef, iou],
        run_eagerly=True
    )

    # weights model path
    checkpoint_path = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.MODEL_CHECKPOINT.PATH,
        f"{cfg.MODEL.WEIGHTS_FILE_NAME}.ckpt"
    )

#     assert os.path.exists(checkpoint_path), \
#         f"Model weight's file does not exist at \n{checkpoint_path}"

    # TODO: verify without augment it produces same results
    # load model weights
    # model.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)
#     with h5py.File(checkpoint_path, 'r', locking=False) as f:
#         model.load_weights(f, by_name=True, skip_mismatch=True)
    model.load_weights(checkpoint_path)
    model.summary()

    # evaluation metric
    # evaluation metric
    evaluation_metrics = ["val_dice_coef", "val_iou"]
    if len(model.outputs) > 1:
        evaluation_metrics = [f"val_{model.output_names[0]}_dice_coef", f"val_{model.output_names[0]}_iou"]

    result = model.evaluate(
        x=val_generator,
        batch_size=cfg.HYPER_PARAMETERS.BATCH_SIZE,
        workers=cfg.DATALOADER_WORKERS,
        return_dict=True,
    )

    # return computed loss, validation accuracy and it's metric name
    return result, evaluation_metric


# -

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Read config file and pass to evaluate method
    """
    result, evaluation_metric = evaluate(cfg)
    print(result)
    print(f"Validation dice coefficient: {result[evaluation_metric]}")


if __name__ == "__main__":
    main()
