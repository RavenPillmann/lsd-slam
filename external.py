import numpy as np

epsil = 1e-6

#
# https://github.com/krrish94/dvo_python/blob/master/se3utils.py
# USED TO TEST MY OWN FUNCTIONS ONLY
def SO3_hat(omega):
	
	omega_hat = np.zeros((3,3))
	omega_hat[0][1] = -omega[2]
	omega_hat[1][0] = omega[2]
	omega_hat[0][2] = omega[1]
	omega_hat[2][0] = -omega[1]
	omega_hat[1][2] = -omega[0]
	omega_hat[2][1] = omega[0]
	
	return omega_hat
	

# SE(3) exponential map
# USED TO TEST MY OWN FUNCTIONS ONLY
def SE3_Exp(xi):
	
	u = xi[:3]
	omega = xi[3:]
	omega_hat = SO3_hat(omega)
	
	theta = np.linalg.norm(omega)
	
	if np.linalg.norm(omega) < epsil:
		R = np.eye(3) + omega_hat
		V = np.eye(3) + omega_hat
	else:
		s = np.sin(theta)
		c = np.cos(theta)
		omega_hat_sq = np.dot(omega_hat, omega_hat)
		theta_sq = theta * theta
		A = s / theta
		B = (1 - c) / (theta_sq)
		C = (1 - A) / (theta_sq)
		R = np.eye(3) + A * omega_hat + B * omega_hat_sq
		V = np.eye(3) + B * omega_hat + C * omega_hat_sq
	t = np.dot(V, u.reshape(3,1))
	lastrow = np.zeros((1,4))
	lastrow[0][3] = 1.
	return np.concatenate((np.concatenate((R, t), axis = 1), lastrow), axis = 0)

# SE(3) Q matrix (according to Tim Barfoot's convention)
# Used in computing the left SE(3) Jacobian
def SE3_Q(xi):
	"""
	This function is to be called only when the axis-angle vector is not very small, as this definition
	DOES NOT take care of small-angle approximations.
	"""
	
	v = xi[:3]
	omega = xi[3:]
	
	theta = np.linalg.norm(omega)
	theta_2 = theta * theta
	theta_3 = theta_2 * theta
	theta_4 = theta_3 * theta
	theta_5 = theta_4 * theta
	
	omega_hat = SO3_hat(omega)
	v_hat = SO3_hat(v)
	
	c = np.cos(theta)
	s = np.sin(theta)
	
	coeff1 = 0.5
	coeff2 = (theta - s) / (theta_3)
	coeff3 = (theta_2 + 2*c - 2) / (2 * theta_4)
	coeff4 = (2*theta - 3*s + theta*c) / (2 * theta_5)
	
	v_hat_omega_hat = np.dot(v_hat, omega_hat)
	omega_hat_v_hat = np.dot(omega_hat, v_hat)
	omega_hat_sq = np.dot(omega_hat, omega_hat)
	omega_hat_v_hat_omega_hat = np.dot(omega_hat, v_hat_omega_hat)
	v_hat_omega_hat_sq = np.dot(v_hat, omega_hat_sq)
	
	matrix1 = v_hat
	matrix2 = omega_hat_v_hat + v_hat_omega_hat + np.dot(omega_hat, v_hat_omega_hat)
	matrix3 = np.dot(omega_hat, omega_hat_v_hat) + v_hat_omega_hat_sq - 3 * omega_hat_v_hat_omega_hat
	matrix4 = np.dot(omega_hat, v_hat_omega_hat_sq) + np.dot(omega_hat, omega_hat_v_hat_omega_hat)
	
	Q = coeff1 * matrix1 + coeff2 * matrix2 + coeff3 * matrix3 + coeff4 * matrix4
	
	return Q


def SE3_left_jacobian(xi):
	
	v = xi[:3]
	omega = xi[3:]
	
	theta = np.linalg.norm(omega)
	xi_curly_hat = SE3_curly_hat(xi)
	
	if theta < epsil:
		return np.eye(6) + 0.5 * xi_curly_hat
	
	J_SO3 = SO3_left_jacobian(omega)
	Q = SE3_Q(xi)
	
	J_SE3 = np.zeros((6,6))
	J_SE3[0:3,0:3] = J_SE3[3:6,3:6] = J_SO3
	J_SE3[0:3,3:6] = Q
	
	return J_SE3

# Return the left jacobian of SO(3)
def SO3_left_jacobian(omega):
	"""
	Takes as input a vector of SO(3) exponential coordinates, and returns the left jacobian of SO(3)
	"""
	
	theta = np.linalg.norm(omega)
	omega_hat = SO3_hat(omega)
	
	if theta < epsil:
		return np.eye(3) + 0.5 * omega_hat
	
	omega_hat_sq = np.dot(omega_hat, omega_hat)
	theta_2 = theta * theta
	theta_3 = theta_2 * theta
	c = np.cos(theta)
	s = np.sin(theta)
	B = (1 - c) / theta_2
	C = (theta - s) / theta_3
	
	return np.eye(3) + B * omega_hat + C * omega_hat_sq


def SE3_curly_hat(xi):
	"""
	Takes in a 6 x 1 vector of SE(3) exponential coordinates and constructs a 6 x 6 adjoint representation.
	(the adjoint in the Lie algebra)
	"""
	v = xi[:3]
	omega = xi[3:]
	xi_curly_hat = np.zeros((6,6)).astype(np.float32)
	omega_hat = SO3_hat(omega)
	xi_curly_hat[0:3,0:3] = omega_hat
	xi_curly_hat[0:3,3:6] = SO3_hat(v)
	xi_curly_hat[3:6,3:6] = omega_hat
	return xi_curly_hat


def clamp(value):
	clamped_value = value
	_min = 0
	_max = 255

	if value < _min:
		clamped_value = _min
	elif value > _max:
		clamped_value = _max

	return clamped_value


# BILINEAR INTERPOLATION from https://github.com/krrish94/dvo_python/blob/master/imgutils.py
def bilinear_interpolation(img, x, y, width, height):

	# Consider the pixel as invalid, to begin with
	valid = np.nan

	# Seems to fail/roll over if x, y are negative because of np.uint8 (unsigned int)
	# TODO: I THINK SOMETHING IS WRONG HERE!!!!!!!!!!!!!!!!
	x0 = np.floor(x).astype(np.uint8)
	y0 = np.floor(y).astype(np.uint8)
	x1 = x0 + 1
	y1 = y0 + 1

	# Compute weights for each corner location, inversely proportional to the distance
	x1_weight = x - x0
	y1_weight = y - y0
	x0_weight = 1 - x1_weight
	y0_weight = 1 - y1_weight

	# Check if the warped points lie within the image
	if x0 < 0 or x0 >= width:
		x0_weight = 0
	if x1 < 0 or x1 >= width:
		x0_weight = 0
	if y0 < 0 or y0 >= height:
		y0_weight = 0
	if y1 < 0 or y1 >= height:
		y1_weight = 0

	# Bilinear weights
	w00 = x0_weight * y0_weight
	w10 = x1_weight * y0_weight
	w01 = x0_weight * y1_weight
	w11 = x1_weight * y1_weight

	# Bilinearly interpolate intensities
	sum_weights = w00 + w10 + w01 + w11
	total = 0
	if w00 > 0:
		total += img.item((y0, x0)) * w00
	if w01 > 0:
		total += img.item((y1, x0)) * w01
	if w10 > 0:
		total += img.item((y0, x1)) * w10
	if w11 > 0:
		total += img.item((y1, x1)) * w11

	if sum_weights > 0:
		valid = total / sum_weights

	# if valid > 255:
	# 	print("x0 y0 x1 y1", x, y, x0, y0, x1, y1)
	# 	print("valid", valid, total, sum_weights)
	# 	print("weights", w00, w01, w10, w11)
	# 	# print("img", img[y0, x0], img[y1, x0], img[y0, x1], img[y1, x1])
	# 	print("x, y", x, y)
	# 	print("x and y weights", x1_weight, y1_weight, x0_weight, y0_weight)

	return valid




# BELOW::: https://github.com/krrish94/dvo_python/blob/master/photometric_alignment.py

# Takes in an intensity image and a registered depth image, and outputs a pointcloud
# Intrinsics must be provided, else we use the TUM RGB-D benchmark defaults.
# https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
def rgbd_to_pointcloud(gray, depth, focal_length, cx, cy, scaling_factor):
	pointcloud = []
	for v in range(gray.shape[1]):
		for u in range(gray.shape[0]):
			intensity = gray.item((u, v))
			Z = depth.item((u, v)) / scaling_factor
			if Z == 0:
				continue
			X = (u - cx) * Z / focal_length
			Y = (v - cy) * Z / focal_length
			pointcloud.append((X, Y, Z, intensity))
	return pointcloud


# Compute photometric error (i.e., the residuals)
def externalComputeResiduals(gray_prev, depth_prev, gray_cur, K, xi):
	"""
	Computes the image alignment error (residuals). Takes in the previous intensity image and 
	first backprojects it to 3D to obtain a pointcloud. This pointcloud is then rotated by an 
	SE(3) transform "xi", and then projected down to the current image. After this step, an 
	intensity interpolation step is performed and we compute the error between the projected 
	image and the actual current intensity image.
	While performing the residuals, also cache information to speedup Jacobian computation.
	"""

	width = gray_cur.shape[1]
	height = gray_cur.shape[0]
	residuals = np.zeros(gray_cur.shape, dtype = np.float32)

	# Cache to store computed 3D points
	cache_point3d = np.zeros((height, width, 3), dtype = np.float32)

	# # Backproject an image to 3D to obtain a pointcloud
	# pointcloud = rgbd_to_pointcloud(gray_prev, depth_prev, focal_length=K['f'], cx=K['cx'], 
	# 	cy=K['cy'], scaling_factor = K['scaling_factor'])
	f_x = K[0, 0]
	f_y = K[1, 1]
	c_x = K[0, 2]
	c_y = K[1, 2]
	
	one_by_f_x = 1. / f_x
	one_by_f_y = 1. / f_y

	# Use the SE(3) Exponential map to compute a 4 x 4 matrix from the vector xi
	T = SE3_Exp(xi)

	# K = np.asarray([[K['f'], 0, K['cx']], [0, K['f'], K['cy']], [0, 0, 1]])
	scaling_factor = 5000 # TODO idk

	# Warp each point in the previous image, to the current image
	for v in range(gray_prev.shape[0]):
		for u in range(gray_prev.shape[1]):
			intensity_prev = gray_prev[v, u]
			Z = depth_prev[v, u] / scaling_factor
			if Z <= 0:
				continue
			Y = one_by_f_y * Z * (v - c_y)
			X = one_by_f_x * Z * (u - c_x)
			# Transform the 3D point
			point_3d = np.dot(T[0:3,0:3], np.asarray([X, Y, Z])) + T[0:3,3]
			point_3d = np.reshape(point_3d, (3,1))
			cache_point3d[v,u,:] = np.reshape(point_3d, (3))
			# Project it down to 2D
			point_2d_warped = np.dot(K, point_3d)
			px = point_2d_warped[0] / point_2d_warped[2]
			py = point_2d_warped[1] / point_2d_warped[2]
			# print("px, u, py, v", px, u, py, v)

			# Interpolate the intensity value bilinearly
			# intensity_warped = np.nan
			# if point_2d_warped[2] > 0 and px[0] < width and py[0] < height:
			intensity_warped = bilinear_interpolation(gray_cur, px[0], py[0], width, height)
			# print("intensity_prev", intensity_prev, intensity_warped, px, py, u, v)

			# If the pixel is valid (i.e., interpolation return a non-NaN value), compute residual
			if not np.isnan(intensity_warped):
				residuals[v, u] = intensity_prev - intensity_warped

	return residuals, cache_point3d


# Function to compute image gradients (used in Jacobian computation)
def externalComputeImageGradients(img):
	"""
	We use a simple form for the image gradient. For instance, a gradient along the X-direction 
	at location (y, x) is computed as I(y, x+1) - I(y, x-1).
	"""
	gradX = np.zeros(img.shape, dtype = np.float32)
	gradY = np.zeros(img.shape, dtype = np.float32)

	width = img.shape[1]
	height = img.shape[0]

	# Exploit the fact that we can perform matrix operations on images, to compute gradients quicker
	gradX[:, 1:width-1] = img[:, 2:] - img[:,0:width-2]
	gradY[1:height-1, :] = img[2:, :] - img[:height-2, :]

	return gradX, gradY


# Compute the Jacobian of the photometric error residual (i.e., the loss function that is 
# being minimized)
def externalComputeJacobian(gray_prev, depth_prev, gray_cur, K, xi, residuals, cache_point3d):
	
	width = gray_prev.shape[1]
	height = gray_prev.shape[0]

	f_x = K[0, 0]
	f_y = K[1, 1]
	c_x = K[0, 2]
	c_y = K[1, 2]

	# Initialize memory to store the Jacobian
	J = np.zeros((height, width, 6))

	# Compute image gradients
	grad_ix, grad_iy = externalComputeImageGradients(gray_cur)

	# For each pixel, compute one Jacobian term
	for v in range(gray_prev.shape[0]):
		for u in range(gray_prev.shape[1]):
			X = cache_point3d[v, u, 0]
			Y = cache_point3d[v, u, 1]
			Z = cache_point3d[v, u, 2]
			if Z <= 0:
				continue
			J_img = np.reshape(np.asarray([[grad_ix[v,u], grad_iy[v,u]]]), (1,2))
			J_pi = np.reshape(np.asarray([[f_x/Z, 0, -f_x*X/(Z*Z)], [0, f_y/2, -f_y*Y/(Z*Z)]]), (2,3))
			J_exp = np.concatenate((np.eye(3), SO3_hat(-np.asarray([X, Y, Z]))), axis=1)
			J_exp = np.dot(J_exp, SE3_left_jacobian(xi))
			J[v,u,:] = residuals[v,u] * np.reshape(np.dot(J_img, np.dot(J_pi, J_exp)), (6))

	return J