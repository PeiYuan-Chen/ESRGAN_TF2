from tensorflow.keras.initializers import VarianceScaling

def scaled_HeNormal(scale_factor=0.2,seed=None):
     return VarianceScaling(scale=2. * scale_factor, mode='fan_in', distribution='truncated_normal', seed=seed)
