import numpy as np
from My_cams.pytorch_grad_cam.base_cam import BaseCAM


class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        print('grads shape ',grads.shape)
        if len(grads.shape) == 2:
            n_channel = grads.shape[1]
            grads = np.swapaxes(grads, 0, 1)
            grads = np.reshape(grads, (n_channel, 8, 8))
            grads = np.expand_dims(grads, axis=0)
        return np.mean(grads, axis=(2, 3))
