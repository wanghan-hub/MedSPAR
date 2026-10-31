"""MedSPAR model exports, without eager heavyweight imports."""
__all__ = ['MedSPAR', 'build_medspar']
def __getattr__(name):
    if name in __all__:
        from . import model
        return getattr(model, name)
    raise AttributeError(name)
