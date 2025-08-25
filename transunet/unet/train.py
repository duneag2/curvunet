"""
Training script
"""
from datetime import datetime
import os
import gc
import hydra
from omegaconf import DictConfig
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    CSVLogger
)

import data_generator
from data_preparation.verify_data import verify_data
from utils.general_utils import create_directory, join_paths, set_gpus, get_gpus_count
from models.model import prepare_model
from losses.loss import dice_coef, iou
from losses.unet_loss import *
import sys

sys.path.append('/home/dragon1/seungeun2025/TransUNet-tf')

# import argparse

# parser = argparse.ArgumentParser(description="UNet3+ Argparse")

# parser.add_argument('--loss_type', type=str, required=True, help='loss type: default, curv')
# parser.add_argument('--model_type', type=str, required=True, help='model type: default, curv, iter')
# parser.add_argument('--curvature', type=str, required=True, help='curvature type: 2d, 3dg, 3dm')
# parser.add_argument('--model_curvature', type=str, required=True, help='model curvature type: 2d, 3dg, 3dm')

# args = parser.parse_args()


def create_training_folders(cfg: DictConfig):
    """
    Create directories to store Model CheckPoint and TensorBoard logs.
    """
    create_directory(
        join_paths(
            cfg.WORK_DIR,
            cfg.CALLBACKS.MODEL_CHECKPOINT.PATH
        )
    )
    create_directory(
        join_paths(
            cfg.WORK_DIR,
            cfg.CALLBACKS.TENSORBOARD.PATH
        )
    )
    
# from tensorflow.keras.callbacks import Callback

# class AdditionalValidationCallback(Callback):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.additional_val_sets = {
#             "val_kits": data_generator.DataGenerator(cfg, mode="VAL_KITS"),
#             "val_anam": data_generator.DataGenerator(cfg, mode="VAL_ANAM"),
#             "val_imperial_pre": data_generator.DataGenerator(cfg, mode="VAL_IMPERIAL_PRE"),
#             "val_imperial_post": data_generator.DataGenerator(cfg, mode="VAL_IMPERIAL_POST"),
#         }

#     def on_epoch_end(self, epoch, logs=None):
#         print("\n================= Additional Validation =================")
#         for name, val_gen in self.additional_val_sets.items():
#             results = self.model.evaluate(val_gen, steps=val_gen.__len__(), return_dict=True, verbose=0)
#             print(f"\n{name} results at epoch {epoch+1}:")
#             for k, v in results.items():
#                 print(f"{k}: {v:.4f}")
#         print("========================================================\n")



def train(cfg: DictConfig):
    """
    Training method
    """
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,3" #0,2 / 1,3


    print("Verifying data ...")
    verify_data(cfg)

    if cfg.MODEL.TYPE == "unet3plus_deepsup_cgm":
        raise ValueError(
            "UNet3+ with Deep Supervision and Classification Guided Module"
            "\nModel exist but training script is not supported for this variant"
            "please choose other variants from config file"
        )

    if cfg.USE_MULTI_GPUS.VALUE:
        # change number of visible gpus for training
        set_gpus(cfg.USE_MULTI_GPUS.GPU_IDS)
        # change batch size according to available gpus
        cfg.HYPER_PARAMETERS.BATCH_SIZE = \
            cfg.HYPER_PARAMETERS.BATCH_SIZE * get_gpus_count()

    # create folders to store training checkpoints and logs
    create_training_folders(cfg)

    # data generators
    train_generator = data_generator.DataGenerator(cfg, mode="TRAIN")
    val_generator = data_generator.DataGenerator(cfg, mode="VAL")
    # val_generator = data_generator.DataGenerator(cfg, mode="VAL_KITS")
    # val_generator = data_generator.DataGenerator(cfg, mode="VAL_ANAM")
    # val_generator = data_generator.DataGenerator(cfg, mode="VAL_IMPERIAL_PRE")
    # val_generator = data_generator.DataGenerator(cfg, mode="VAL_IMPERIAL_POST")

    # verify generator
    # for i, (batch_images, batch_mask) in enumerate(val_generator):
    #     print(len(batch_images))
    #     if i >= 3: break

    # optimizer
    # TODO update optimizer
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
            model = prepare_model(cfg, True, cfg.model_type, cfg.model_curvature)
    else:
        model = prepare_model(cfg, True, cfg.model_type, cfg.model_curvature)
    
    if cfg.loss_type == 'default':
        chosen_loss = unet3p_hybrid_loss
    elif cfg.loss_type == 'curv':
        if cfg.curvature == '2d':
            chosen_loss = unet3p_hybrid_loss_2d
        elif cfg.curvature == '3dg':
            chosen_loss = unet3p_hybrid_loss_3dg
        elif cfg.curvature == '3dm':
            chosen_loss = unet3p_hybrid_loss_3dm
    print('chosen loss: ', chosen_loss)
    model.compile(
        optimizer=optimizer,
        loss=chosen_loss,
        metrics=[dice_coef, iou],
        run_eagerly=True #False - 이유는 모르겠지만 fused만 False이고 default와 curv는 True이다...
    )
    model.summary()

    # the tensorboard log directory will be a unique subdirectory
    # based on the start time for the run
    tb_log_dir = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.TENSORBOARD.PATH,
        "{}".format(datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
    )
    print("TensorBoard directory\n" + tb_log_dir)

    checkpoint_path = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.MODEL_CHECKPOINT.PATH,
        f"{cfg.MODEL.WEIGHTS_FILE_NAME}.ckpt"
    )
    print("Weights path\n" + checkpoint_path)

    csv_log_path = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.CSV_LOGGER.PATH,
        f"training_logs_{cfg.MODEL.TYPE}.csv"
    )
    print("Logs path\n" + csv_log_path)

    # print('model output: ', model.outputs)
    # evaluation metric
    evaluation_metrics = ["val_dice_coef", "val_iou"]
    if len(model.outputs) > 1:
        evaluation_metrics = [f"val_{model.output_names[0]}_dice_coef", f"val_{model.output_names[0]}_iou"]


    # TensorBoard, EarlyStopping, ModelCheckpoint and CSVLogger callbacks
#     callbacks = [
#         TensorBoard(log_dir=tb_log_dir, write_graph=False, profile_batch=0),
#         EarlyStopping(
#             monitor=evaluation_metrics[1],
#             patience=cfg.CALLBACKS.EARLY_STOPPING.PATIENCE,
#             verbose=cfg.VERBOSE
#         ),
#         ModelCheckpoint(
#             checkpoint_path,
#             verbose=cfg.VERBOSE,
#             save_weights_only=cfg.CALLBACKS.MODEL_CHECKPOINT.SAVE_WEIGHTS_ONLY,
#             save_best_only=cfg.CALLBACKS.MODEL_CHECKPOINT.SAVE_BEST_ONLY,
#             monitor=evaluation_metrics[1],
#             mode="max",
#             save_format='tf'

#         ),
#         CSVLogger(
#             csv_log_path,
#             append=cfg.CALLBACKS.CSV_LOGGER.APPEND_LOGS
#         ),
# #         AdditionalValidationCallback(cfg),
#     ]
    callbacks = [
    TensorBoard(log_dir=tb_log_dir, write_graph=False, profile_batch=0),
    # EarlyStopping 제거하거나 아래처럼 바꿔서 사용:
    # EarlyStopping(monitor="loss", patience=cfg.CALLBACKS.EARLY_STOPPING.PATIENCE, verbose=cfg.VERBOSE),

    ModelCheckpoint(
        checkpoint_path,
        verbose=cfg.VERBOSE,
        save_weights_only=cfg.CALLBACKS.MODEL_CHECKPOINT.SAVE_WEIGHTS_ONLY,
        save_best_only=cfg.CALLBACKS.MODEL_CHECKPOINT.SAVE_BEST_ONLY,
        monitor="loss",
        mode="min",
        save_format='tf'
    ),
    CSVLogger(
        csv_log_path,
        append=cfg.CALLBACKS.CSV_LOGGER.APPEND_LOGS
    ),
    # AdditionalValidationCallback 제거
    ]


    training_steps = train_generator.__len__()
    validation_steps = val_generator.__len__()

    # start training
    model.fit(
        x=train_generator,
        steps_per_epoch=training_steps,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=cfg.HYPER_PARAMETERS.EPOCHS,
        batch_size=cfg.HYPER_PARAMETERS.BATCH_SIZE,
        callbacks=callbacks,
        workers=cfg.DATALOADER_WORKERS,
    )
    
    # print("\n================ FINAL VALIDATION =================")
    # val_sets = {
    #     # "val_main": val_generator,
    #     "val_kits": data_generator.DataGenerator(cfg, mode="VAL_KITS"),
    #     "val_anam": data_generator.DataGenerator(cfg, mode="VAL_ANAM"),
    #     "val_imperial_pre": data_generator.DataGenerator(cfg, mode="VAL_IMPERIAL_PRE"),
    #     "val_imperial_post": data_generator.DataGenerator(cfg, mode="VAL_IMPERIAL_POST"),
    # }

    # for name, val_gen in val_sets.items():
    #     print(f"\nEvaluating on {name} dataset...")
    #     results = model.evaluate(val_gen, steps=val_gen.__len__(), return_dict=True, verbose=0)# , verbose=0
    #     print(f"{name} results:")
    #     for k, v in results.items():
    #         print(f"{k}: {v:.4f}")
            
    #     del val_gen
    #     gc.collect()
    #     tf.keras.backend.clear_session()  # GPU memory
    # print("====================================================\n")

    
    
    
#     # ========== 추가 validation 데이터셋 평가 ==========
#     additional_val_sets = {
#         "val_kits": data_generator.DataGenerator(cfg, mode="VAL_KITS"),
#         "val_anam": data_generator.DataGenerator(cfg, mode="VAL_ANAM"),
#         "val_imperial_pre": data_generator.DataGenerator(cfg, mode="VAL_IMPERIAL_PRE"),
#         "val_imperial_post": data_generator.DataGenerator(cfg, mode="VAL_IMPERIAL_POST"),
#     }

#     for name, val_gen in additional_val_sets.items():
#         print(f"\nEvaluating on {name} dataset...")
#         results = model.evaluate(val_gen, steps=val_gen.__len__(), return_dict=True)
#         print(f"{name} results:")
#         for k, v in results.items():
#             print(f"{k}: {v:.4f}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Read config file and pass to train method for training
    """
    train(cfg)


if __name__ == "__main__":
    main()
