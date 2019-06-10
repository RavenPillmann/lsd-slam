import numpy as np
import cv2
import math
from external import bilinear_interpolation

# https://jsturm.de/publications/data/engel2013iccv.pdf
# ALTHOUGH MY IMPLEMENTATION is based on the original paper, I referenced an implementation of depth estimation
# for additional information. Some of my reasoning reflects that.
# https://github.com/geoeo/Prob_Depth/blob/master/depth_estimation.py
MINIMUM_ABS_GRADIENT_VALUE_FOR_REFINING = 1
MINIMUM_ABS_GRADIENT_VALUE_FOR_CREATING = 1

ANGLE_THRESHOLD = 0.3
# TODO CHANGE THIS MAYBE
MINIMUM_EPIPOLAR_GRADIENT_SQUARED = 2

MAXIMUM_INVERSE_DEPTH = 1 / 0.05
MINIMUM_INVERSE_DEPTH = 0

MAXIMUM_INCREMENT_LENGTH = 30

MAX_MATCHING_ERROR = 4 * 1300.0

CAMERA_PIXEL_NOISE = 4.0 * 4.0

DIVISION_EPS = 1e-10

DEBUG = False

def printline(*args):
	args = [str(arg) for arg in args]
	if DEBUG:
		print(" ".join(args))


# https://github.com/geoeo/Prob_Depth/blob/master/libs/frame.py
def interpolate(_map, _x, _y):
	# try:
	return cv2.getRectSubPix(_map, (1, 1), (_x, _y))[0][0]
	# except:
		# print("interpolate() error", _map.shape, _x, _y)



def subPixelInterpolate(best_error, best_match_error_pre, best_match_error_pre_difference, best_match_error_post, best_match_error_post_difference, best_match_x, best_match_y, increment_x, increment_y):
	try:
		gradient_pre_pre = -(best_match_error_pre - best_match_error_pre_difference)
		gradient_pre_curr = best_error - best_match_error_pre_difference
		gradient_post_curr = -(best_error - best_match_error_post_difference)
		gradient_post_post = best_match_error_post - best_match_error_post_difference
	except:
		print("subPixelInterpolate() error", best_match_error_pre, best_match_error_pre_difference, best_error, best_match_error_post_difference, best_match_error_post)


	if ((gradient_pre_pre < 0) ^ (gradient_pre_curr < 0)) and not ((gradient_post_post < 0) ^ (gradient_post_curr < 0)):
		_d = gradient_pre_curr / (gradient_pre_curr - gradient_pre_pre)
		best_match_x -= _d * increment_x
		best_match_y -= _d * increment_y
		best_error -= 2 * _d * gradient_pre_curr - (gradient_pre_pre - gradient_pre_curr) * np.square(_d)

	elif ((gradient_post_post < 0) ^ (gradient_post_curr < 0)):
		_d = gradient_post_curr / (gradient_post_curr - gradient_post_post)
		best_match_x += _d * increment_x
		best_match_y += _d * increment_y
		best_error += 2 * _d * gradient_post_curr - (gradient_post_post - gradient_post_curr) * np.square(_d)

	return best_match_x, best_match_y, best_error


def clamp(value):
	clamped_value = value
	_min = -1e-10
	_max = 1e-10

	if value < 0 and value > _min:
		clamped_value = _min
	elif value < _max:
		clamped_value = _max

	return clamped_value


def getSubsetOfPixels(image, keyframe, K, extrinsic_matrix, image_dimensions):
	"""
	Input: 	image (R ^(m x n)) we are warping to
			keyframe (R ^(m x n)) we are warping from (this is the reference frame)
			K (uninverted)
			extrinsic_matrix (from keyframe to image)
	Output:	list of pixel locations in this image that are "good"
	"""
	# Accuracy of disparity search needs to be sufficiently large
	# 3 Criteria used:
		# 1) Photometric disparity error
		# 2) Geometric disparity error
		# 3) Pixel to inverse depth ratio (which seems to bring in the other two?)
	
	# TODO
	# Go through pixels,
	# 	Check Photometric, Geometric, Pixel Ratio

	# Based on https://github.com/geoeo/Prob_Depth/blob/master/depth_estimation.py, there seems to be another
	# way of finding good pixels that, as far as I can tell, is equivalent to this. 

	keyframe_gradient_x = keyframe.getGradientX()
	keyframe_gradient_y = keyframe.getGradientY()
	keyframe_current_inv_depth_map = keyframe.getInverseDepthMap()

	# gradient = np.array([keyframe_gradient_x, keyframe_gradient_y])

	# Keyframe is where points will come from
	abs_gradient = np.sqrt(np.square(keyframe_gradient_x) + np.square(keyframe_gradient_y))

	_X, _Y = np.meshgrid(range(image.shape[1]), range(image.shape[0]))

	X_potentially_good = None
	Y_potentially_good = None

	# if keyframe.getIsNew():
	X_potentially_good = _X[abs_gradient >= MINIMUM_ABS_GRADIENT_VALUE_FOR_CREATING]
	Y_potentially_good = _Y[abs_gradient >= MINIMUM_ABS_GRADIENT_VALUE_FOR_CREATING]
	# else:
		# X_potentially_good = _X[abs_gradient >= MINIMUM_ABS_GRADIENT_VALUE_FOR_REFINING]
		# Y_potentially_good = _Y[abs_gradient >= MINIMUM_ABS_GRADIENT_VALUE_FOR_REFINING]


	(_successes, _epipolar_x, _epipolar_y, _good_x, _good_y) = getGoodPixels(X_potentially_good, Y_potentially_good, np.linalg.inv(K), extrinsic_matrix, image_dimensions, keyframe_gradient_x, keyframe_gradient_y)

	good_pixels = []

	number_of_successes = 0

	if _successes.any():
		for success, epx, epy, x, y in zip(_successes, _epipolar_x, _epipolar_y, _good_x, _good_y):
			  number_of_successes = number_of_successes + 1
			  good_pixels.append((x, y, epx, epy))

	printline("getSubsetOfPixels() number_of_successes", number_of_successes)

	return good_pixels


def getGoodPixels(X_potentially_good, Y_potentially_good, K_inv, extrinsic_matrix, image_dimensions, keyframe_gradient_x, keyframe_gradient_y):
	fundamental_matrix = buildFundamentalMatrix(K_inv, extrinsic_matrix)

	# TODO (continue with this https://github.com/geoeo/Prob_Depth/blob/master/depth_estimation.py#L306)
	stacked = np.vstack((X_potentially_good, Y_potentially_good, np.ones(X_potentially_good.shape)))
	# epipolar = np.matmul(fundamental_matrix, stacked)
	epipolar = np.matmul(fundamental_matrix.T, stacked)

	# epipolar will be lines on one image. Stacked should therefore be points on the other image.
	# Right now points are in keyframe, epipolar lines are in other image...supposedly.
	# As far as I can tell, the fundamental matrix is then from image to keyframe (recause rotation/translation are inverted before F is formed)


	# print("epipolar", epipolar.shape)
	epipolar_x_fund = epipolar[1, :] / image_dimensions[1]
	# epipolar_x = np.array(epipolar_x_fund[0].flat)
	epipolar_x = np.array(epipolar_x_fund.flat)


	epipolar_y_fund = -epipolar[0, :] / image_dimensions[0]
	# epipolar_y = np.array(epipolar_y_fund[0].flat)
	epipolar_y = np.array(epipolar_y_fund.flat)

	# print("epx fund", epipolar_x_fund)
	# print("epx", epipolar_x)

	epipolar_magnitude = np.sqrt(np.square(epipolar_x) + np.square(epipolar_y))
	# print("epx epy", np.square(epipolar_x) + np.square(epipolar_y))

	# print("epipolar_magnitude", epipolar_magnitude.shape)

	epipolar_gradient_x = keyframe_gradient_x[Y_potentially_good, X_potentially_good] * epipolar_x
	epipolar_gradient_y = keyframe_gradient_y[Y_potentially_good, X_potentially_good] * epipolar_y

	epipolar_gradient_squared_and_normalized = np.square(epipolar_gradient_x + epipolar_gradient_y) / np.square(epipolar_magnitude)

	squared_absolute_gradient = np.square(keyframe_gradient_x[Y_potentially_good, X_potentially_good]) + np.square(keyframe_gradient_y[Y_potentially_good, X_potentially_good])
	angle = epipolar_gradient_squared_and_normalized / squared_absolute_gradient

	gradient_norm = np.sqrt(squared_absolute_gradient)

	# TODO: gradient_sample_distribution
	# fac = 1.0 / epipolar_magnitude

	# print("getGoodPixels() epipolar_magnitude", epipolar_magnitude ** 2)
	# print("getGoodPixels() ~np.isnan(epipolar_x + epipolar_y))", ~np.isnan(epipolar_x + epipolar_y))
	# print("getGoodPixels() epipolar_gradient_squared_and_normalized", epipolar_gradient_squared_and_normalized)

	# Finally, doing checks here that are equivalent to the 3 conditions layed out in the depth estimate paper
	# TODO: need angle threshold

	# print("epipolar magnitude", epipolar_magnitude)
	epipolar_magnitude_squared = epipolar_magnitude ** 2
	# print("ep mag squared 1", epipolar_magnitude_squared)


	# TODO: 
	# is_epipolar_mag_squared_big_enough = False
	if (np.isnan(np.sum(epipolar_magnitude_squared))):
		# nan_values = np.isnan(epipolar_magnitude_squared)
		# nan_indices = np.where(nan_values)
		# epipolar_magnitude_squared[nan_indices] = 0.0
		epipolar_magnitude_squared = np.nan_to_num(epipolar_magnitude_squared)
		# is_epipolar_mag_squared_big_enough = True if ((epipolar_magnitude_squared) >= 1e-6) else False 

	# is_epipolar_gradient_squared_and_normalized_greater_than_thresh = False
	if (np.isnan(np.sum(epipolar_gradient_squared_and_normalized))):
		# nan_values = np.isnan(epipolar_gradient_squared_and_normalized)
		# nan_indices = np.where(nan_values)
		# epipolar_gradient_squared_and_normalized[nan_indices] = 0.0
		epipolar_gradient_squared_and_normalized = np.nan_to_num(epipolar_gradient_squared_and_normalized)

	# print(angle)
	# is_angle_large_enough = False
	if (np.isnan(np.sum(angle))):
		# nan_values = np.isnan(angle)
		# nan_indices = np.where(nan_values)
		# angle[nan_indices] = 0.0
		angle = np.nan_to_num(angle)
		# is_angle_large_enough = (angle > ANGLE_THRESHOLD)
	# print(angle)

	# print("Validate", np.isnan(np.sum(epipolar_magnitude_squared)), np.isnan(np.sum(epipolar_gradient_squared_and_normalized)), np.isnan(np.sum(angle)))
	# print("angle", angle > ANGLE_THRESHOLD)
	# print("epi grad", epipolar_gradient_squared_and_normalized >= MINIMUM_EPIPOLAR_GRADIENT_SQUARED)
	
	# TODO: vvv WHY are these both scalars???
	# print("eg mg squared", epipolar_magnitude_squared)
	# print("epi x and y", ~np.isnan(epipolar_x + epipolar_y))
	# print("ep mg squared >= 1e-6", (epipolar_magnitude_squared) >= 1e-6)

	good_indices = np.where((angle > ANGLE_THRESHOLD) &
		(epipolar_gradient_squared_and_normalized >= MINIMUM_EPIPOLAR_GRADIENT_SQUARED) &
		(~np.isnan(epipolar_x + epipolar_y)) & 
		((epipolar_magnitude_squared) >= 1e-6))[0]

	good_indices = np.where((angle > ANGLE_THRESHOLD) &
		(epipolar_gradient_squared_and_normalized >= MINIMUM_EPIPOLAR_GRADIENT_SQUARED))[0]

	# print("getGoodPixels() good_indices", good_indices)
	# print(~np.isnan(epipolar_x + epipolar_y))
	# print(((epipolar_magnitude_squared) >= 1e-6))

	# print("getGoodPixels() past the indices finder")

	epipolar_line_x_normalized = np.zeros(X_potentially_good.shape)
	epipolar_line_y_normalized = np.zeros(Y_potentially_good.shape)
	succeeded = np.full_like(X_potentially_good, False).astype(bool)

	# print(set(good_indices))
	epipolar_line_x_normalized = np.array(epipolar_x[good_indices]) * np.array(1.0 / epipolar_magnitude[good_indices])
	epipolar_line_y_normalized = np.array(epipolar_y[good_indices]) * np.array(1.0 / epipolar_magnitude[good_indices])
	succeeded[good_indices] = True

	return (succeeded[good_indices], epipolar_line_x_normalized, epipolar_line_y_normalized, X_potentially_good[good_indices], Y_potentially_good[good_indices])


def invertExtrinsicMatrix(extrinsic_matrix):
	# Need to implement inversion
	rotation_matrix = extrinsic_matrix[0:3, 0:3]
	rotation_transpose = rotation_matrix.T

	translation = extrinsic_matrix[0:3, 3].reshape(3, 1)
	inverse_translation = -np.matmul(rotation_transpose, translation)

	# Inverse of [R t] is [R_inv -R_inv*t]
	inverse_matrix = np.concatenate((rotation_transpose, inverse_translation), axis=1)
	inverse_matrix = np.concatenate((inverse_matrix, np.array([[0, 0, 0, 1]])), axis=0)

	return inverse_matrix


def buildEssentialMatrix(extrinsic_matrix):
	rotation = extrinsic_matrix[0:3, 0:3]
	translation = extrinsic_matrix[0:3, 3]
	translation_skew_symmetric = np.zeros((3, 3))
	tz = translation[2]
	ty = translation[1]
	tx = translation[0]

	translation_skew_symmetric[0, 1] = -tz
	translation_skew_symmetric[0, 2] = ty
	translation_skew_symmetric[1, 0] = tz
	translation_skew_symmetric[1, 2] = -tx
	translation_skew_symmetric[2, 0] = -ty
	translation_skew_symmetric[2, 1] = tx

	return np.matmul(translation_skew_symmetric, rotation)


def buildFundamentalMatrix(intrinsic_matrix_inverse, extrinsic_matrix):
	"""
	Input: 		intrinsic_matrix_inverse (inv(K))
				extrinsic_matrix (This is assumed to be the transformation from keyframe to current frame)
	Output: 	3x3 fundamental matrix
	"""
	inverse_extrinsic_transformation = invertExtrinsicMatrix(extrinsic_matrix)
	# inverse_transformation = inverse_extrinsic_transformation[0:3, 3]

	essential_matrix = buildEssentialMatrix(inverse_extrinsic_transformation)

	fundamental_matrix = np.matmul(np.matmul(intrinsic_matrix_inverse.T, essential_matrix), intrinsic_matrix_inverse)


	# t_x = inverse_extrinsic_transformation[0, 3]
	# t_y = inverse_extrinsic_transformation[1, 3]
	# t_z = inverse_extrinsic_transformation[2, 3]
	# K = np.linalg.inv(intrinsic_matrix_inverse)
	# f_x = K[0, 0]
	# f_y = K[1, 1]
	# c_x = K[0, 2]
	# c_y = K[1, 2]
	# fundamental_matrix = np.zeros((3, 3))
	# fundamental_matrix[0, 0] = 0
 #    fundamental_matrix[1, 0] = t_z
 #    fundamental_matrix[2, 0] = -f_y * t_y - c_y * t_z
 #    fundamental_matrix[0, 1] = -t_z
 #    fundamental_matrix[1, 1] = 0
 #    fundamental_matrix[2, 1] = f_x * t_x + c_x * t_z
 #    fundamental_matrix[0, 2] = f_y * t_y + c_y * t_z
 #    fundamental_matrix[1, 2] = -f_x * t_x - c_x * t_z
 #    fundamental_matrix[2, 2] = 0

	return fundamental_matrix


# TODO: DO I NEED THIS TO INSTEAD FLIP THE U ANV V???? I think maybe
def getValuesToSearchForInImage(keyframe_image, u_keyframe, v_keyframe, epipolar_line_x_normalized, epipolar_line_y_normalized, rescale_factor, flip_epipole_point_to_other_side):
	search_value_one = interpolate(
		keyframe_image,
		u_keyframe + 2 * epipolar_line_x_normalized * rescale_factor * flip_epipole_point_to_other_side,
		v_keyframe + 2 * epipolar_line_y_normalized * rescale_factor * flip_epipole_point_to_other_side
	)

	search_value_one = bilinear_interpolation(
		keyframe_image,
		v_keyframe + 2 * epipolar_line_x_normalized * rescale_factor * flip_epipole_point_to_other_side,
		u_keyframe + 2 * epipolar_line_y_normalized * rescale_factor * flip_epipole_point_to_other_side,
		keyframe_image.shape[1],
		keyframe_image.shape[0]
	)

	search_value_two = interpolate(
		keyframe_image,
		u_keyframe + epipolar_line_x_normalized * rescale_factor * flip_epipole_point_to_other_side,
		v_keyframe + epipolar_line_y_normalized * rescale_factor * flip_epipole_point_to_other_side
	)

	search_value_two = bilinear_interpolation(
		keyframe_image,
		v_keyframe + epipolar_line_x_normalized * rescale_factor * flip_epipole_point_to_other_side,
		u_keyframe + epipolar_line_y_normalized * rescale_factor * flip_epipole_point_to_other_side,
		keyframe_image.shape[1],
		keyframe_image.shape[0]
	)

	search_value_three = interpolate(
		keyframe_image,
		u_keyframe,
		v_keyframe
	)

	search_value_three = bilinear_interpolation(
		keyframe_image,
		v_keyframe,
		u_keyframe,
		keyframe_image.shape[1],
		keyframe_image.shape[0]
	)

	search_value_four = interpolate(
		keyframe_image,
		u_keyframe - epipolar_line_x_normalized * rescale_factor * flip_epipole_point_to_other_side,
		v_keyframe - epipolar_line_y_normalized * rescale_factor * flip_epipole_point_to_other_side
	)

	search_value_four = bilinear_interpolation(
		keyframe_image,
		v_keyframe - epipolar_line_x_normalized * rescale_factor * flip_epipole_point_to_other_side,
		u_keyframe - epipolar_line_y_normalized * rescale_factor * flip_epipole_point_to_other_side,
		keyframe_image.shape[1],
		keyframe_image.shape[0]
	)

	search_value_five = interpolate(
		keyframe_image,
		u_keyframe - 2 * epipolar_line_x_normalized * rescale_factor * flip_epipole_point_to_other_side,
		v_keyframe - 2 * epipolar_line_y_normalized * rescale_factor * flip_epipole_point_to_other_side
	)

	search_value_five = bilinear_interpolation(
		keyframe_image,
		v_keyframe - 2 * epipolar_line_x_normalized * rescale_factor * flip_epipole_point_to_other_side,
		u_keyframe - 2 * epipolar_line_y_normalized * rescale_factor * flip_epipole_point_to_other_side,
		keyframe_image.shape[1],
		keyframe_image.shape[0]
	)

	return search_value_one, search_value_two, search_value_three, search_value_four, search_value_five


def findBestMatch(image, current_point_x, current_point_y, increment_x, increment_y, point_close_x, point_close_y, keyframe_search_points, image_search_points, rescale_factor):
	"""
	Input: 		keyframe_search_points, image_search_points - these should be 1x5
	"""
	succeeded = True
	iteration = 0
	best_loop = -1
	second_best_loop = -1
	best_match_x = -1
	best_match_y = -1

	best_error = float('Inf')
	second_best_error = float('Inf')

	best_match_error_pre = float('Inf')
	best_match_error_pre_difference = float('Inf')
	best_match_error_post = float('Inf')
	best_match_error_post_difference = float('Inf')

	ee_last = -1

	error_a = np.array([0, 0, 0, 0, 0])
	error_b = np.array([0, 0, 0, 0, 0])

	while (((increment_x < 0) == (current_point_x > point_close_x)) and ((increment_y < 0) == (current_point_y > point_close_y))):
		image_search_points[4] = interpolate(image, current_point_x + 2. * increment_x, current_point_y + 2. * increment_y)
		iteration_error = 0.

		if iteration % 2 == 0:
			error_a = np.abs(image_search_points - keyframe_search_points)
			iteration_error = np.dot(error_a, error_a.T)
		else:
			error_b = np.abs(image_search_points - keyframe_search_points)
			iteration_error = np.dot(error_b, error_b.T)

		if (iteration_error < best_error):
			second_best_error = best_error
			second_best_loop = best_loop
			best_error = iteration_error
			best_loop = iteration

			best_match_x = current_point_x
			best_match_y = current_point_y

			best_match_error_pre = ee_last
			best_match_error_pre_difference = np.dot(error_a, error_b.T)
			best_match_error_post = -1
			best_match_error_post_difference = -1

		elif (iteration_error < second_best_error):
			second_best_error = iteration_error
			second_best_loop = iteration

		ee_last = iteration_error

		image_search_points[0] = image_search_points[1]
		image_search_points[1] = image_search_points[2]
		image_search_points[2] = image_search_points[3]
		image_search_points[3] = image_search_points[4]

		current_point_x += increment_x
		current_point_y += increment_y

		iteration += 1

	# best error is still too large
	if ((best_error > MAX_MATCHING_ERROR) or
		((abs(best_loop - second_best_loop) > 1) and ((1.5 * best_error) > second_best_error))): 
		succeeded = False

	best_match_x, best_match_y, best_error = subPixelInterpolate(best_error, best_match_error_pre, best_match_error_pre_difference, best_match_error_post, best_match_error_post_difference, best_match_x, best_match_y, increment_x, increment_y)
	
	sample_distance = 1.0 * rescale_factor
	gradient_along_line = 0

	for i in range(1, 5):
		gradient_along_line = (image_search_points[i] - image_search_points[i - 1]) ** 2

	gradient_along_line /= (sample_distance ** 2)

	threshold = 1300.0 + np.sqrt(gradient_along_line) * 20.
	succeeded = best_error <= threshold

	return succeeded, best_match_x, best_match_y, sample_distance, gradient_along_line


def getInverseDepth(extrinsic_matrix_keyframe_to_image, u_keyframe, v_keyframe, K, K_inv, keyframe, image, epipolar_line_x_normalized, epipolar_line_y_normalized, previous_inverse_depth, maximum_inverse_depth, minimum_inverse_depth, image_dimensions):

	epipolar_scale = 2 # NOT SURE IF THIS IS GOOD AT ALL
	# Error codes
	error_in_calc_search_space = (False, -1, 0, 0, 0, 0, 0, 0, 0, (u_keyframe, v_keyframe, 1))
	error_in_matching = (False, -1, 0, 0, 0, 0, 0, 0, 0, (u_keyframe, v_keyframe, 2))
	error_in_depth_computation = (False, -1, 0, 0, 0, 0, 0, 0, 0, (u_keyframe, v_keyframe, 3))
	# Get the pixel in the image, use K_inv to get the pixel in real-world
	#
	# Normalized image coordinates of point in keyframe image
	point = np.array([u_keyframe, v_keyframe, 1]).T
	real_world_point = np.matmul(K_inv, point)
	if (real_world_point[2] != 1):
		print("real_world_point", real_world_point)

	(succeeded, point_close, point_far, increment_x, increment_y,
		rescale_factor, increment_length, flip_epipole_point_to_other_side) = calculateSearchSpaceInImageFrame(
		image,
		keyframe,
		K,
		K_inv,
		extrinsic_matrix_keyframe_to_image,
		real_world_point,
		previous_inverse_depth,
		maximum_inverse_depth,
		minimum_inverse_depth,
		epipolar_scale,
		image_dimensions
	)

	if not succeeded:
		return error_in_calc_search_space

	keyframe_image = keyframe.getCameraImage()

	# Pick five points along the epipolar line
	# Should be using interpolation, just floating for now

	keyframe_search_one, keyframe_search_two, keyframe_search_three, keyframe_search_four, keyframe_search_five = getValuesToSearchForInImage(keyframe_image, u_keyframe, v_keyframe, epipolar_line_x_normalized, epipolar_line_y_normalized, rescale_factor, flip_epipole_point_to_other_side)

	point_far_x = point_far[0]
	point_far_y = point_far[1]

	image_search_one = bilinear_interpolation(image, point_far_x - 2. * increment_x, point_far_y - 2. * increment_y, image.shape[1], image.shape[0])
	image_search_two = bilinear_interpolation(image, point_far_x - increment_x, point_far_y - increment_y, image.shape[1], image.shape[0])
	image_search_three = bilinear_interpolation(image, point_far_x, point_far_y, image.shape[1], image.shape[0])
	image_search_four = bilinear_interpolation(image, point_far_x + increment_x, point_far_y + increment_y, image.shape[1], image.shape[0])
	image_search_five = bilinear_interpolation(image, point_far_x + 2. * increment_x, point_far_y + 2 * increment_y, image.shape[1], image.shape[0])

	keyframe_search_points = np.array([keyframe_search_one, keyframe_search_two, keyframe_search_three, keyframe_search_four, keyframe_search_five])
	image_search_points = np.array([image_search_one, image_search_two, image_search_three, image_search_four, image_search_five])

	# print("point_close", point_close)
	point_close_x = point_close[0]
	point_close_y = point_close[1]
	succeeded, best_match_x, best_match_y, sample_distance, gradient_along_line = findBestMatch(image, point_far_x, point_far_y, increment_x, increment_y, point_close[0], point_close[1], keyframe_search_points, image_search_points, rescale_factor)

	if not succeeded:
		return error_in_matching

	(succeeded, new_inverse_depth, alpha, large_x_disparity) = calculateInverseDepthInKeyframe(extrinsic_matrix_keyframe_to_image, increment_x, increment_y, K_inv, best_match_x, best_match_y, v_keyframe, u_keyframe, K)

	if not succeeded:
		return error_in_depth_computation

	new_variance = calculateVariance(keyframe, gradient_along_line, u_keyframe, v_keyframe, epipolar_line_x_normalized, epipolar_line_y_normalized, alpha, sample_distance)

	# TODO update this (SHOULD LAST VALUE BE SOMETHING ELSE? 10 or 20 as other code suggests?)
	return (True, new_inverse_depth, new_variance, increment_length, best_match_x, best_match_y, point_close, point_far, rescale_factor, (u_keyframe, v_keyframe, large_x_disparity))


def calculateVariance(keyframe, gradient_along_line, keyframe_u, keyframe_v, epipolar_x_norm, epipolar_y_norm, alpha, sample_distance):
	printline("calculateVariance()")
	photometric_disparity_error = 4. * CAMERA_PIXEL_NOISE / (gradient_along_line + DIVISION_EPS)
	tracking_error = 0.25
	# gradient_x = interpolate(keyframe.getGradientX(), keyframe_u, keyframe_v)
	# gradient_y = interpolate(keyframe.getGradientY(), keyframe_u, keyframe_v)

	gradient_x = bilinear_interpolation(keyframe.getGradientX(), keyframe_v, keyframe_u, keyframe.getGradientX().shape[1], keyframe.getGradientX().shape[0])
	gradient_y = bilinear_interpolation(keyframe.getGradientY(), keyframe_v, keyframe_u, keyframe.getGradientY().shape[1], keyframe.getGradientY().shape[0])
	# try:
	# print("norm(epipolar_x_norm)", np.sqrt(np.dot(epipolar_x_norm.T, epipolar_x_norm)), np.sqrt(np.dot(epipolar_y_norm.T, epipolar_y_norm)))
	geometric_disparity_error = gradient_x * epipolar_x_norm + gradient_y * epipolar_y_norm + DIVISION_EPS
	# except:
		# print("calculateVariance() error", gradient_x, epipolar_x_norm, gradient_y, epipolar_y_norm)
	geometric_disparity_error = np.square(tracking_error) * (
			np.square(gradient_x) + np.square(gradient_y)) / (np.square(geometric_disparity_error))

	# TODO: I think I need to square this to make it variance
	return np.square(alpha * alpha * (0.05 * np.square(sample_distance) + geometric_disparity_error + photometric_disparity_error))


def calculateInverseDepthInKeyframe(extrinsic_matrix_keyframe_to_image, increment_x, increment_y, K_inv, best_match_x, best_match_y, keyframe_x, keyframe_y, K):
	# Depth = f*Baseline / Disparity, so inv_depth = disparity/f*baseline
	# Now looking back at paper, check against other implementation
	# https://github.com/geoeo/lsd_slam/blob/master/lsd_slam_core/src/DepthEstimation/DepthMap.cpp#L1878
	rotation = extrinsic_matrix_keyframe_to_image[0:3, 0:3]
	translation = extrinsic_matrix_keyframe_to_image[0:3, 3]

	fxi = K_inv[0, 0]
	fyi = K_inv[1, 1]
	cxi = K_inv[0, 2]
	cyi = K_inv[1, 2]

	fx = K[0, 0]
	fy = K[1, 1]
	cx = K[0, 2]
	cy = K[1, 2]

	keyframe_point_in_real_world = np.dot(K_inv, np.array([keyframe_x, keyframe_y, 1]).T)
	# print("keyframe_point_in_real_world", keyframe_point_in_real_world)

	new_inverse_depth = -1
	alpha = -1
	large_x_disparity = False

	# print("increment_x, increment_y", abs(increment_x), abs(increment_y))

	if (abs(increment_x) > abs(increment_y)):
		large_x_disparity = True
		old_x = fxi * best_match_x + cxi
		# denominator = old_x * translation[2] - translation[0]

		dot_0 = np.dot(keyframe_point_in_real_world, rotation[0, :])
		dot_2 = np.dot(keyframe_point_in_real_world, rotation[2, :])

		new_inverse_depth = (dot_0 - old_x * dot_2) / denominator
		alpha = increment_x * fxi * (dot_0 * translation[2] - dot_2 * translation[0]) / np.square(denominator)

		base_line = translation[0] - keyframe_point_in_real_world[0] * translation[2]
		denominator = fx * translation[0] - cx * translation[2]


		dot_0 = np.dot(rotation[0, :], keyframe_point_in_real_world)
		dot_2 = np.dot(rotation[2, :], keyframe_point_in_real_world)

		x_rotation = keyframe_point_in_real_world[0] * dot_2
		x_displacement = dot_0 - x_rotation

		new_inverse_depth = x_displacement / base_line

		numerator = best_match_x - cx * dot_2 - fx * dot_0

		alpha = increment_x * fxi * (dot_0 * translation[2] - dot_2 * translation[0]) / (base_line * base_line)
	else:
		large_x_disparity = False
		old_y = fyi * best_match_y + cyi;
		denominator = old_y * translation[2] - translation[0]
		dot_1 = np.dot(keyframe_point_in_real_world, rotation[1, :])
		dot_2 = np.dot(keyframe_point_in_real_world, rotation[2, :])

		new_inverse_depth = (dot_1 - old_y * dot_2) / denominator;
		alpha = increment_y * fyi * (dot_1 * translation[2] - dot_2 * translation[1]) / np.square(denominator)
		# vvvvv

		base_line = translation[1] - keyframe_point_in_real_world[1] * translation[2]

		dot_1 = np.dot(rotation[1, :], keyframe_point_in_real_world)
		dot_2 = np.dot(rotation[2, :], keyframe_point_in_real_world)
		y_rotation = keyframe_point_in_real_world[1] * dot_2

		y_displacement = dot_1 - y_rotation
		
		inverse_depth_new = y_displacement / base_line

		denom = fy*translation[1] + cy*translation[2]
		numerator = best_match_y -cy*dot_2 - fy*dot_1

		inner_1 = (dot_1 * translation[2] - dot_2 * translation[1])
		alpha = increment_y * fxi * inner_1 / (base_line * base_line)

	if (new_inverse_depth < 0):
		return False, -1, -1, False

	if new_inverse_depth == 0:
		new_inverse_depth = 0.001

	return True, new_inverse_depth, alpha, large_x_disparity


def calculateSearchSpaceInImageFrame(image, keyframe, K, K_inv, extrinsic_matrix_keyframe_to_image, real_world_point, previous_inverse_depth, maximum_inverse_depth, minimum_inverse_depth, epipolar_scale, image_dimensions):
	"""
	Input:		real_world_point - pixel in real world
				extrinsic_matrix_keyframe_to_image - Keyframe to image rotation/translation
	"""
	error = (False, 0, 0, 0, 0, 0, 0, 0)


	rotation = extrinsic_matrix_keyframe_to_image[0:3, 0:3]
	translation = extrinsic_matrix_keyframe_to_image[0:3, 3]
	k_rotation = np.matmul(K, rotation)
	k_translation = np.matmul(K, translation)
	p_inf = np.matmul(k_rotation, real_world_point)

	key_to_image_translation_x = translation[0]
	key_to_image_translation_y = translation[1]
	key_to_image_translation_z = translation[2]

	flip_epipole_point_to_other_side = 1.

	# If not mainly a z translation, need to flip epipolar point 
	if not (np.square(key_to_image_translation_z) > np.square(key_to_image_translation_x) and
		np.square(key_to_image_translation_z) > np.square(key_to_image_translation_y)):
		flip_epipole_point_to_other_side = -1.

	point_in_image_frame = p_inf + k_translation * previous_inverse_depth
	# print("calculateSearchSpaceInImageFrame()", point_in_image_frame, p_inf, k_translation, previous_inverse_depth)

	# Get z as the rescale factor
	rescale_factor = point_in_image_frame[2]

	# Safeguard that seems to ignore points that change too much
	# if DEBUG:
	if (rescale_factor <= 0.7 or rescale_factor >= 1.4):
		printline('calculateSearchSpaceInImageFrame() failed because of rescale_factor: ', rescale_factor)
		return error

	maximum_translation = k_translation * maximum_inverse_depth * flip_epipole_point_to_other_side
	point_close_real = p_inf + maximum_translation

	# Normalize the scale

	# VVV
	# if (point_close_real[2] < 0.001):
	# 	maximum_inverse_depth = (0.001 - p_inf[2]) / k_translation[2]
	# 	point_close_real = p_inf + k_translation * maximum_inverse_depth

	# ^^^

	point_close = point_close_real / point_close_real[2]

	minimum_translation = k_translation * minimum_inverse_depth
	point_far_real = p_inf + minimum_translation

	if point_close[2] < 0.001 or maximum_inverse_depth < minimum_inverse_depth:
		printline("calculateSearchSpaceInImageFrame() failed because of point_close, max, or min inverse depth", point_close, maximum_inverse_depth, minimum_inverse_depth)
		return error


	point_far = point_far_real / point_far_real[2]

	if np.isnan(point_far[0] + point_close[0]):
		printline("calculateSearchSpaceInImageFrame() failed because something is nan", point_far, point_close)
		return error

	increment = point_close - point_far
	increment_x = increment[0]
	increment_y = increment[1]

	increment_length = np.sqrt(np.square(increment_x) + np.square(increment_y))
	if increment_length <= 0 or np.isnan(increment_length):
		printline("calculateSearchSpaceInImageFrame() failed because increment_length", increment_length)
		return error

	if increment_length > MAXIMUM_INCREMENT_LENGTH:
		point_close[0] = point_far[0] + increment_x * MAXIMUM_INCREMENT_LENGTH / increment_length
		point_close[1] = point_far[1] + increment_y * MAXIMUM_INCREMENT_LENGTH / increment_length

	increment_x *= 1. / increment_length
	increment_y *= 1. / increment_length

	if epipolar_scale != 0.0:
		point_far[0] -= epipolar_scale * increment_x
		point_far[1] -= epipolar_scale * increment_y
		point_close[0] += 1. * increment_x
		point_close[1] += 1. * increment_y

	point_to_border = 7

	# end here if point is outside of image and cannot be back projected as a result
	if (point_far[0] <= point_to_border or
		point_far[0] >= image_dimensions[1] - point_to_border or
		point_far[1] <= point_to_border or
		point_far[1] >= image_dimensions[0] - point_to_border):
		printline("calculateSearchSpaceInImageFrame() failed because point_far, point_to_border, or image_dimensions", point_far, point_to_border, image_dimensions)
		return error

	if (point_close[0] <= point_to_border or
		point_close[0] >= image_dimensions[1] - point_to_border or
		point_close[1] <= point_to_border or
		point_close[1] >= image_dimensions[0] - point_to_border):
		printline("calculateSearchSpaceInImageFrame() failed because point_close, point_to_border, or image_dimensions", point_close, point_to_border, image_dimensions)

		return error

	return (True, point_close, point_far, increment_x, increment_y, rescale_factor, increment_length, flip_epipole_point_to_other_side)


def createInverseDepthMap(keyframe, u_keyframe, v_keyframe, epipolar_line_x_normalized, epipolar_line_y_normalized, extrinsic_matrix_keyframe_to_image, K, K_inv, image):
	# print("createInverseDepthMap()")
	min_inverse_depth = 0.00001
	max_inverse_depth = 1 / 0.05
	# prior_inverse_depth = 1.0
	prior_inverse_depth = keyframe.getInverseDepthPixel(u_keyframe, v_keyframe)

	(succeeded, new_inverse_depth, new_variance, increment_length,
		best_match_x, best_match_y, point_close, point_far, rescale_factor, keyframe_point) = getInverseDepth(
		extrinsic_matrix_keyframe_to_image, u_keyframe, v_keyframe,
		K, K_inv, keyframe, image, epipolar_line_x_normalized, epipolar_line_y_normalized, prior_inverse_depth, max_inverse_depth, min_inverse_depth, image.shape)

	if not succeeded:
		printline("getInverseDepth() Failed", keyframe_point)
		return (False, (0, 0, 0, 0), 0, 0, keyframe_point, rescale_factor)

	clamped_depth = clamp(new_inverse_depth)

	keyframe.setInverseDepthPixel(u_keyframe, v_keyframe, new_inverse_depth)
	keyframe.setInverseDepthVariancePixel(u_keyframe, v_keyframe, new_variance)

	return (succeeded, (u_keyframe, v_keyframe, best_match_y, best_match_x), point_close, point_far, keyframe_point, rescale_factor)


def updateInverseDepthMap(keyframe, u_keyframe, v_keyframe, epipolar_line_x_normalized, epipolar_line_y_normalized, extrinsic_matrix_keyframe_to_image, K, K_inv, image):
	# print("updateInverseDepthMap()")
	current_keyframe_inverse_depth = keyframe.getInverseDepthPixel(u_keyframe, v_keyframe)
	current_keyframe_inverse_variance = keyframe.getInverseDepthVariancePixel(u_keyframe, v_keyframe)
	current_keyframe_inverse_stddev = np.sqrt(current_keyframe_inverse_variance)

	# Compute the new depth and variance by using the kalman filter updates found in engle 2013

	# Set the new restrictions of where to look
	min_inverse_depth = current_keyframe_inverse_depth - 2. * current_keyframe_inverse_stddev
	max_inverse_depth = current_keyframe_inverse_depth + 2. * current_keyframe_inverse_stddev

	min_inverse_depth = max(0, min_inverse_depth)
	max_inverse_depth = min(20, max_inverse_depth)

	(succeeded, inverse_depth, variance, increment_length, best_match_x, best_match_y,
	point_close, point_far, rescale_factor, keyframe_point) = getInverseDepth(
		extrinsic_matrix_keyframe_to_image, u_keyframe, v_keyframe,
		K, K_inv, keyframe, image, epipolar_line_x_normalized, epipolar_line_y_normalized, current_keyframe_inverse_depth, max_inverse_depth, min_inverse_depth, image.shape)

	if not succeeded:
		 return (False, (0, 0, 0, 0), 0, 0, keyframe_point, rescale_factor)

	# Seems like we should increase the current variance a little bit to deal with some sort of noise
	adjusted_current_inverse_variance = 1.01 * current_keyframe_inverse_variance
	new_variance = (adjusted_current_inverse_variance * variance) / (adjusted_current_inverse_variance + variance)
	new_inverse_depth = (adjusted_current_inverse_variance * inverse_depth + variance * current_keyframe_inverse_depth) / (adjusted_current_inverse_variance + variance)

	if new_variance < current_keyframe_inverse_variance:
		keyframe.setInverseDepthVariancePixel(u_keyframe, v_keyframe, new_variance)

	printline("updateInverseDepthMap", new_inverse_depth)
	keyframe.setInverseDepthPixel(u_keyframe, v_keyframe, new_inverse_depth)

	return (succeeded, (u_keyframe, v_keyframe, best_match_y, best_match_x), point_close, point_far, keyframe_point, rescale_factor)


def estimateDepthMap(image, keyframe, K, extrinsic_matrix, image_dimensions):
	"""
	Input:		image - image being warped to
				keyframe - keyframe with depthmap being refined
				K - intrinsic camera parameters
				extrinsic_matrix - extrinsic camera parameters (keyframe to image)
				image_dimensions - dimension of image (height, width)
	"""
	# Three overall steps:
	# 1) get subset of pixels for which accuracy of a disparity is sufficiently large
	# 2) For each pixel, select a suitable reference frame, perform 1D disparity search
	# 3) Obtained depth map is fused into depth map (?)

	good_pixels = getSubsetOfPixels(image, keyframe, K, extrinsic_matrix, image_dimensions)
	matches = []
	matches_large_x_disparity = []
	keypoints = []

	K_inv = np.linalg.inv(K)

	update_count = 0
	for pixel in good_pixels:
		keyframe_v = pixel[0] # keyframe_v refers to x-axis
		keyframe_u = pixel[1] # keyframe_u refers to y-axis
		epipolar_line_x_normalized = pixel[2]
		epipolar_line_y_normalized = pixel[3]
		if keyframe.getIsNew(keyframe_u, keyframe_v):
			# Create a depth map
			if pixel == good_pixels[0]:
				printline("createInverseDepthMap()")
			(succeeded, match_keyframe_image, point_close_image, point_far_image, keyframe_point, rescale_factor) = createInverseDepthMap(keyframe, keyframe_u, keyframe_v, epipolar_line_x_normalized, epipolar_line_y_normalized, extrinsic_matrix, K, K_inv, image)
			if pixel == good_pixels[0] and not succeeded:
				printline("Not succeeded")
		else:
			# Refine a depth map
			# if pixel == good_pixels[0]:
			printline("updateInverseDepthMap()")
			update_count += 1
			(succeeded, match_keyframe_image, point_close_image, point_far_image, keyframe_point, rescale_factor) = updateInverseDepthMap(keyframe, keyframe_u, keyframe_v, epipolar_line_x_normalized, epipolar_line_y_normalized, extrinsic_matrix, K, K_inv, image)

		keyframe_x = keyframe_point[0]
		keyframe_y = keyframe_point[1]
		large_x_disparity = keyframe_point[2]

		if succeeded:
			(x, y, best_match_x, best_match_y) = match_keyframe_image

			matches.append((x, y, best_match_x, best_match_y, epipolar_line_x_normalized, epipolar_line_y_normalized, point_close_image, point_far_image, rescale_factor))
			if large_x_disparity:
				matches_large_x_disparity.append((x, y, best_match_x, best_match_y, epipolar_line_x_normalized, epipolar_line_y_normalized, point_close_image, point_far_image))

		keypoints.append(keyframe_point)
	# print("!!!Number of keypoints", len(keypoints))

	# TODO: Now have the option of denoising, depth_map_smoothing

	print("Updated", update_count, "times")
	return (good_pixels, matches, matches_large_x_disparity, keypoints)


def main():
	pass


if __name__ == "__main__":
	main()