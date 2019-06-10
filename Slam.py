import numpy as np
import cv2
import sys

from Extract import readFrames
from Warping import warpImage, expSe3Mapping, logSe3Mapping, poseConcatenation
from Keyframe import Keyframe
from DepthMap import estimateDepthMap, buildFundamentalMatrix
from external import bilinear_interpolation

from plyfile import PlyData, PlyElement

DEBUG = True

NEW_KEYFRAME_THRESHOLD = 1e-7 # IDK


def shouldCreateNewKeyframe(xi):
	print("dist new keyframe", np.matmul(xi.T, xi))
	if np.matmul(xi.T, xi) > NEW_KEYFRAME_THRESHOLD:
		return True

	return False


def testFundamental(K, rotation_translation_old_xi, original_image, current_image):
	fund = buildFundamentalMatrix(np.linalg.inv(K), rotation_translation_old_xi)

	epipolar_line_sample_1 = np.array([
		[360],
		[222],
		[1]
	])

	epipolar_line_sample_2 = np.array([
		[200],
		[250],
		[1]
	])

	epipolar_line_sample_3 = np.array([
		[250],
		[400],
		[1]
	])

	ep_1 = np.matmul(fund, epipolar_line_sample_1)
	ep_2 = np.matmul(fund, epipolar_line_sample_2)
	ep_3 = np.matmul(fund, epipolar_line_sample_3)
	ep_1 = np.matmul(epipolar_line_sample_1.T, fund).T
	ep_2 = np.matmul(epipolar_line_sample_2.T, fund).T
	ep_3 = np.matmul(epipolar_line_sample_3.T, fund).T

	print("Ep_samp_1", ep_1)
	print("Ep_samp_2", ep_2)

	# TODO: How to draw these with opencv on image??
	# previous_image, current_image
	# y = (ax + c) / b
	y_1_1 = -(ep_1[0] * 0 + ep_1[2])/ep_1[1]
	y_1_100 = -(ep_1[0] * 1000 + ep_1[2])/ep_1[1]

	print("y11", y_1_1)
	print("y1100", y_1_100)

	y_2_1 = -(ep_2[0] * 0 + ep_2[2]) / ep_2[1]
	y_2_100 = -(ep_2[0] * 1000 + ep_2[2]) / ep_2[1]

	print("y21", y_2_1)
	print("y2100", y_2_100)

	y_3_1 = -(ep_3[0] * 0 + ep_3[2]) / ep_3[1]
	y_3_100 = -(ep_3[0] * 1000 + ep_3[2]) / ep_3[1]

	print("y21", y_3_1)
	print("y2100", y_3_100)

	cv2.line(original_image, (0, y_1_1), (1000, y_1_100), (255, 0, 0), 2)
	cv2.line(original_image, (0, y_2_1), (1000, y_2_100), (0, 255, 0), 2)
	cv2.line(original_image, (0, y_3_1), (1000, y_3_100), (0, 0, 255), 2)
	# cv2.imshow("image", current_image)

	cv2.circle(current_image, (360, 222), 2, (0, 0, 255), -1)
	cv2.circle(current_image, (200, 250), 2, (0, 0, 255), -1)
	cv2.circle(current_image, (250, 400), 2, (0, 0, 255), -1)

	horz_image = np.concatenate((original_image, current_image), axis=1)
	cv2.imshow("horz_image", horz_image)

	cv2.waitKey(10000)
	cv2.destroyAllWindows()
	cv2.waitKey(1)

	print(rotation_translation_old_xi)


def rescaleInverseDepth(keyframe):
	# Before moving to next keyframe, need to rescale
	inverse_depth = keyframe.getInverseDepthMap()
	mean_inv_depth = np.mean(inverse_depth)

	rescaled_inverse_depth = inverse_depth / mean_inv_depth
	keyframe.setInverseDepthMap(rescaled_inverse_depth)


def export3DPoints(keyframe, K_inv):
	inverse_depth = keyframe.getInverseDepthMap()
	# depth = 1 / inverse_depth

	points = []

	for i in range(inverse_depth.shape[0]):
		for j in range(inverse_depth.shape[1]):
			pixel_coordinate = np.array([
				[j],
				[i],
				[1],
			])

			real_world_coordinate = np.matmul(K_inv, pixel_coordinate) / inverse_depth[i, j]

			points.append((real_world_coordinate[0][0], real_world_coordinate[1][0], real_world_coordinate[2][0]))

	points = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
	# print(points.shape)

	element = PlyElement.describe(points, 'real_world_coordinates')

	PlyData([element]).write('coordinates_10_rescaled_inv_depth.ply')


def main():
	# TODO: test rotation, depth estimation
	# So first get rotation with a random depth estimation
	# Then call the depth estimation
	# This will not work at first, need to debug heavily
	images, K = readFrames(sys.argv[1], sys.argv[2])
	print("K", K)
	previous_image = images[0]
	current_image = images[1]
	# cv2.imshow("prev", previous_image)
	# cv2.waitKey(5000)
	# cv2.destroyAllWindows()

	# Random values for inverse depth and variance, initially
	inverse_depth = np.abs(np.random.uniform(0, 1e6, previous_image.shape)) + 1e3
	inverse_variance = 1e6 * np.ones(previous_image.shape)

	# inverse_depth = np.ones(previous_image.shape)
	# inverse_variance = np.ones(previous_image.shape)

	# Create the first keyframe

	initial_camera_position = np.array([
		[1., 0., 0., 0.],
		[0., 1., 0., 0.],
		[0., 0., 1., 0.],
		[0., 0., 0., 1.]
	])
	keyframe = Keyframe(previous_image, inverse_depth, inverse_variance, initial_camera_position)

	rotation_translation_matrix = np.array([
		[ 0.91390406, -0.37368099, -0.15856195, 10.29721872],
		[ 0.07217044,  0.53396537, -0.84242056, -2.62571743],
		[ 0.39946314,  0.75844809,  0.51496184,  0.57883017],
		[ 0.,          0.,          0.,          1.        ]
	])

	original_image = keyframe.getCameraImage()
	previous_image = keyframe.getCameraImage()
	old_xi = warpImage(current_image, previous_image, K, inverse_depth, keyframe)
	# rotation_translation_old_xi = expSe3Mapping(old_xi)
	# fund = buildFundamentalMatrix(np.linalg.inv(K), rotation_translation_old_xi)

	# print("KEYFRAME Mean INV DEPTH VARIANCE", np.max(keyframe.getInverseDepthVariance()), np.mean(keyframe.getInverseDepthVariance()), np.min(keyframe.getInverseDepthVariance()))
	# print("KEYFRAME Mean INV DEPTH", np.max(keyframe.getInverseDepthMap()), np.mean(keyframe.getInverseDepthMap()), np.min(keyframe.getInverseDepthMap()))

	estimateDepthMap(current_image, keyframe, K, expSe3Mapping(old_xi), previous_image.shape)

	# print("KEYFRAME Mean INV DEPTH VARIANCE", np.max(keyframe.getInverseDepthVariance()), np.mean(keyframe.getInverseDepthVariance()), np.min(keyframe.getInverseDepthVariance()))
	# print("KEYFRAME Mean INV DEPTH", np.max(keyframe.getInverseDepthMap()), np.mean(keyframe.getInverseDepthMap()), np.min(keyframe.getInverseDepthMap()))

	for i in range(2, 3):
		print("At index", i)
		previous_image = images[i - 1]
		current_image = images[i]
		xi = warpImage(current_image, previous_image, K, keyframe.getInverseDepthMap(), keyframe)
		# rotation_translation_xi = expSe3Mapping(xi)
		# rotation_translation_old_xi = np.matmul(rotation_translation_old_xi, rotation_translation_xi)
		# rotation_translation_old_xi = np.matmul(rotation_translation_xi, rotation_translation_old_xi)
		old_xi = poseConcatenation(xi, old_xi)

		# TODO: I don't think this is correct order for pose concatenation
		# Alt: Feed initial xi to warpImage, let warpImage be the change from keyframe to current image, rather than previous image to current image???
		# old_xi = poseConcatenation(old_xi, xi)
		shallCreateNewKeyframe = shouldCreateNewKeyframe(old_xi)
		if shallCreateNewKeyframe:
			print("Create New Keyframe", i)

		estimateDepthMap(current_image, keyframe, K, expSe3Mapping(old_xi), previous_image.shape)
		print("KEYFRAME Mean INV DEPTH VARIANCE", np.max(keyframe.getInverseDepthVariance()), np.mean(keyframe.getInverseDepthVariance()), np.min(keyframe.getInverseDepthVariance()))
		print("KEYFRAME Mean INV DEPTH", np.max(keyframe.getInverseDepthMap()), np.mean(keyframe.getInverseDepthMap()), np.min(keyframe.getInverseDepthMap()))

	testFundamental(K, expSe3Mapping(old_xi), original_image, current_image)

	rescaleInverseDepth(keyframe)
	export3DPoints(keyframe, np.linalg.inv(K))



if __name__ == "__main__":
	main()