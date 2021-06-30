from tensor2tensor.utils import hparam
import tensorflow as tf

from text import symbols


def create_hparams(hparams_string=None, verbose=False):

    hparams = hparam.HParams(
        seed=1234,

        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],
        # ignore_layers=None,

        # Settings for training with WaveGlow
        prefix_data='_WAVEGLOW/',
        extension_data='WAVEGLOW',
        dim_data=80,
        # fe_data=22050.0 / 275,    # ~80Hz in line with WaveRNN
        fe_data=22050.0 / 256,      # ~86Hz in line with WAVEGLOW

        nm_csv_train='csv_files/ES_LMP_NEB_01_0001.csv',
        nm_csv_test='csv_files/ES_LMP_NEB_01_0001.csv',
        lg_data_max=1700,

        with_end_silence=True,
        end_silence_ms=100.0,
        loss_gate_weight=1.0,
        verbose_logs=False,
        checkpoint_freq=1,
 
        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1700,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet=True,
        # Desactivate postnet at start
        start_without_postnet=True,
        epoch_start_postnet=10, # postnet=False before epoch_start_postnet, et True if i_epoch >= epoch_start_postnet
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=True,
        momentum=0.5,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=8, #it was 64
        nb_epochs=100,
        offset_epoch=0,
        mask_padding=True,  # set model's padded outputs to padded values

        # Decreasing learning rate
        fixed_learning_rate=False,
        # if fixed_learning_rate=True
        learning_rate=1e-5,
        # if fixed_learning_rate=False
        learning_rate_max=1e-3,
        learning_rate_min=1e-5,
        epoch_start=10,
        dr=0.6,
        
        ################################
        # Save data                    #
        ################################
        save_embeddings=True,
        save_alignments=True
    )

    if hparams_string:
#        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
