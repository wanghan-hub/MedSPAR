"""TC-MVE visual interface and complete Stage1 training model."""
__all__ = ['TC_MVE', 'SteerMedSAM', 'build_steer_medsam', 'build_stage0_medsam']
def __getattr__(name):
    if name == 'TC_MVE':
        from .encoder import TC_MVE
        return TC_MVE
    if name in __all__:
        from . import model
        return getattr(model, name)
    raise AttributeError(name)
