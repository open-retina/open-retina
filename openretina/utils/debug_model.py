import numpy as np
import torch


def is_model_causal(model, length_stimulus_same: int = 40, length_stimulus_different: int = 20) -> bool:
    """Check if a given model is causal or not
    That is if the response of a neural gets influenced by stimulus changes in the future (noncausal) or not (causal)"""
    model_was_training = model.training
    model.eval()

    total_time = length_stimulus_same + length_stimulus_different
    fixed = torch.ones(model.stimulus_shape(total_time), device=model.device)
    with torch.no_grad():
        response = model.forward(fixed)[0].detach().cpu().numpy()

    time_delay = total_time - response.shape[0]

    stimulus_same = torch.ones(model.stimulus_shape(length_stimulus_same), device=model.device)
    stimulus_different = 2.0 * torch.ones(model.stimulus_shape(length_stimulus_different), device=model.device)
    concat_stimulus = torch.cat([stimulus_same, stimulus_different], dim=2)

    with torch.no_grad():
        response_concat = model.forward(concat_stimulus)[0].detach().cpu().numpy()

    time_point_when_response_should_differ = length_stimulus_same - time_delay
    first_time_point_different_response = int(np.argmax(response_concat[:, 0] != response[:, 0]))

    is_model_causal = time_point_when_response_should_differ <= first_time_point_different_response

    # reset model in its original train or eval state
    model.train(model_was_training)

    return is_model_causal
