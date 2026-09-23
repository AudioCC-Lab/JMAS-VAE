import json

def create_model_from_config(model_config):
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'

    if model_type == 'autoencoder':
        from .autoencoders import create_autoencoder_from_config
        return create_autoencoder_from_config(model_config)
    else:
        raise NotImplementedError(f'Unsupported model type for JMAS-VAE release: {model_type}')

def create_model_from_config_path(model_config_path):
    with open(model_config_path) as f:
        model_config = json.load(f)
    
    return create_model_from_config(model_config)

def create_pretransform_from_config(pretransform_config, sample_rate):
    raise NotImplementedError('Pretransforms are not part of the JMAS-VAE release configs')

def create_bottleneck_from_config(bottleneck_config):
    bottleneck_type = bottleneck_config.get('type', None)

    assert bottleneck_type is not None, 'type must be specified in bottleneck config'

    if bottleneck_type == 'vae':
        from .bottleneck import VAEBottleneck
        bottleneck = VAEBottleneck()
    elif bottleneck_type == 'jmasvae_ssl_new':
        from .bottleneck import JMASVAESSL_new_Bottleneck
        bottleneck = JMASVAESSL_new_Bottleneck(**bottleneck_config.get('config', {}))
    else:
        raise NotImplementedError(f'Unsupported bottleneck type for JMAS-VAE release: {bottleneck_type}')
    
    requires_grad = bottleneck_config.get('requires_grad', True)
    if not requires_grad:
        for param in bottleneck.parameters():
            param.requires_grad = False

    return bottleneck
