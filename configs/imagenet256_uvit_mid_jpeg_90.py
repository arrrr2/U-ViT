import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.z_shape = (4, 32, 32)

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth'
    )

    config.train = d(
        n_steps=400000,
        batch_size=256,
        mode='cond',
        log_interval=100,
        eval_interval=100000,
        save_interval=200000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='uvit',
        img_size=32,
        patch_size=2,
        in_chans=4,
        embed_dim=768,
        depth=16,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=1001,
        use_checkpoint=False
    )

    config.dataset = d(
        name='imagenet256_features_lmdb',
        path='/home/ubuntu/data/datasets/imagenet256_latents/jpeg/90/',
        cfg=True,
        p_uncond=0.1
    )

    config.sample = d(
        sample_steps=50,
        n_samples=50000,
        mini_batch_size=24,  # the decoder is large
        algorithm='dpm_solver',
        cfg=True,
        scale=0.4,
        path=''
    )

    config.lossy_compression = d(
        codec='jpeg',
        quality_fact='90',
        dataset_portion='1.0',
    )

    return config
