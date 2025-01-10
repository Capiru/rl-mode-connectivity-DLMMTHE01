from mode_connectivity.interpolate import interpolate_models
from alphago.utils import get_model
import torch

def test_interpolation():
    model_1 = get_model()
    
    model_2 = get_model()
    
    interpolated_model_0 = interpolate_models(model_1=model_1,model_2=model_2,alpha=0.0)
    interpolated_model_05 = interpolate_models(model_1=model_1,model_2=model_2,alpha=0.5)
    interpolated_model_1 = interpolate_models(model_1=model_1,model_2=model_2,alpha=1.0)
    
    params_model_1 = [param.clone() for param in model_1.parameters()]
    params_model_2 = [param.clone() for param in model_2.parameters()]
    params_int_0 = [param.clone() for param in interpolated_model_0.parameters()]
    params_int_05 = [param.clone() for param in interpolated_model_05.parameters()]
    params_int_1 = [param.clone() for param in interpolated_model_1.parameters()]
    diff_weights = False
    for param1, param2,param_0,param_05,param_1 in zip(params_model_1, params_model_2, params_int_0, params_int_05,params_int_1):
        # When alpha = 0, must be equal to param2
        if not torch.equal(param2,param_0):
            diff_weights = True
            break
        # When alpha = 1, must be equal to param1
        if not torch.equal(param1,param_1):
            diff_weights = True
            break
        # When alpha = 0.5 cannot be equal to either
        if torch.equal(param1,param_05) and torch.equal(param2,param_05):
            diff_weights = True
            break
        assert torch.equal(param_05,0.5*param1+0.5*param2)
    assert not diff_weights