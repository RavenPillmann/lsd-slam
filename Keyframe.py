# Raven Pillmann
import numpy as np

class Keyframe:
	camera_image = np.array()
	inverse_depth_map = np.array()
	variance_inverse_depth = np.array()

	def __init__(self, camera_image=np.array(), inverse_depth_map=np.array(), variance_inverse_depth=np.array()):
    	self.camera_image = camera_image
    	self.inverse_depth_map = inverse_depth_map
    	self.variance_inverse_depth = variance_inverse_depth