# This file is covered by the license agreement found in the file "license.txt" in the root of this project.
from .model_pms import *

_supported_models = ['E2EModelEM',"E2EModel_deep", "E2EModelPMS"] 

def is_supported(model_name):
    if model_name in _supported_models: return True


class CustomDataParallel(nn.DataParallel):
    """Multi GPU support for models with custom attributes"""
    def __init__(self, module, custom_props=[], *args, **kwargs):
        super().__init__(module, *args, **kwargs)
        self._custom_props = custom_props

    def __getattr__(self, name):
        if name in self._custom_props:
            return getattr(self.module, name)
        else:
            return super().__getattr__(name)

class ModelFactory:
    """A factory class to create VCM compatible models. This class is only for cleaner script and better code readability from the design pattern point of view. The function(s) are @staticmethod, which means they can be seen just like normal module functions (no class binding). 
    """
    @staticmethod 
    def create_model(args) -> tuple:
        """Create model by interpreting the model name and args
            Returns the model instance and the input size divisible requirement in a tuple
        """
        model_name = args.model
        assert is_supported(model_name), f"Invalid argument {model_name}. Supported models: {_supported_models}."
        pms_profile = args.probmodel.pms_profile
        
        if model_name in globals():
            size_divisible = 32
            if model_name == "E2EModelPMS":
                if pms_profile == "baseline":
                    size_divisible = 8
                elif pms_profile == "extra":
                    size_divisible = 8
            model = globals()[model_name](args)
            model.size_divisible = size_divisible
            return model