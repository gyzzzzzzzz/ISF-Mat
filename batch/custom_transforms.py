from mmcv.transforms import BaseTransform, TRANSFORMS
import numpy as np

@TRANSFORMS.register_module()
class BrightnessMultiplier(BaseTransform):
    def __init__(self, multiplier=1.0):
        self.multiplier = multiplier

    def transform(self, results):
        img = results['img'].astype(np.float32) * self.multiplier
        results['img'] = np.clip(img, 0, 255).astype(np.uint8)
        return results