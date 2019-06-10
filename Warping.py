import numpy as np
import cv2
import sys

from Extract import readFrames
from external import bilinear_interpolation, externalComputeJacobian, externalComputeImageGradients, externalComputeResiduals

# http://www.dis.uniroma1.it/~labrococo/tutorial_icra_2016/icra16_slam_tutorial_engel.pdf

def logSe3Mapping(mat):
	"""
	Input: mat = 4x4 matrix
	Output: xi = 6x1 twist
	"""
	R = mat[0:3, 0:3]
	t = mat[0:3, 3]
	t = t.reshape((3, 1))
	theta = np.arccos((np.trace(R) - 1) / 2)
	ln_R = (theta / (2*np.sin(theta))) * (R - R.T)

	# Not sure but is this the next step?
	omega = np.array([ln_R[2][1], ln_R[0][2], ln_R[1][0]])

	theta = np.sqrt(np.dot(omega, omega.T))
	A = np.sin(theta) / theta
	B = (1 - np.cos(theta)) / (theta**2)

	V_inv = np.eye(3) - .5 * ln_R + (1 / (theta**2))*(1 - (A / (2*B))) * (np.dot(ln_R, ln_R))

	u = np.matmul(V_inv, t)

	xi = np.array([
		[u[0,0]],
		[u[1,0]],
		[u[2,0]],
		[omega[0]],
		[omega[1]],
		[omega[2]]
	])

	return xi


def poseConcatenation(element_1, element_2):
	"""
	Input: 	element_1 = 6x1 twist
			element_2 = 6x1 twist
	Output:	val = 6x1 twist
	"""
	exp_se3_1 = expSe3Mapping(element_1)
	exp_se3_2 = expSe3Mapping(element_2)

	return logSe3Mapping(np.matmul(exp_se3_1, exp_se3_2))


def projectPoint(point):
	return np.array([
		[point[0, 0] / point[2, 0]],
		[point[1, 0] / point[2, 0]],
		[1 / point[2, 0]]
	])


def getRotationFromXi(xi):
	rotation_x = xi[3]
	rotation_y = xi[4]
	rotation_z = xi[5]

	rotation_matrix = np.array([
		[0, -rotation_z, rotation_y],
		[rotation_z, 0, -rotation_x],
		[-rotation_y, rotation_x, 0]
	])

	return rotation_matrix


# Is this right????
def getPhotometricVariance(keyframe, jacobian):
	inverse_variance_2d = keyframe.getInverseDepthVariance()
	covariance_matrix_inverse_variance = np.zeros((inverse_variance_2d.shape[0] * inverse_variance_2d.shape[1], inverse_variance_2d.shape[0] * inverse_variance_2d.shape[1]))

	for i in range(inverse_variance_2d.shape[0]):
		for j in range(inverse_variance_2d.shape[1]):

			covariance_matrix_inverse_variance[j * inverse_variance_2d.shape[0] + i][0] = inverse_variance_2d[i, j]

	# return np.matmul(jacobian, np.matmul(covariance_matrix_inverse_variance, jacobian.T))
	return -1


# TODO: Test
def getDepthResidual(keyframe, unprojected_image_points):
	depth_residuals = np.sum(unprojected_image_points - keyframe.getInverseDepthMap())
	return


def getTranslationFromXi(xi):
	pass


def expSe3Mapping(xi):
	# Apply se(3): http://ethaneade.com/lie_groups.pdf
	rotation_x = xi[3, 0]
	rotation_y = xi[4, 0]
	rotation_z = xi[5, 0]
	translation_x = xi[0, 0]
	translation_y = xi[1, 0]
	translation_z = xi[2, 0]

	omega = np.array([
		[rotation_x],
		[rotation_y],
		[rotation_z]
	])

	omega_cross_matrix = np.array([
		[0, 			-rotation_z, 		rotation_y],
		[rotation_z, 			0, 			-rotation_x],
		[-rotation_y,		rotation_x, 			0]
	])

	translation_vector = np.array([
		[translation_x],
		[translation_y],
		[translation_z]
	])

	theta = np.sqrt(np.dot(omega.T, omega))

	R = np.zeros(3)
	V = np.zeros(3)

	# print("theta", theta)

	if theta < 1e-6:
		R = np.eye(3) + omega_cross_matrix
		V = np.eye(3) + omega_cross_matrix

	else:
	# I BELIEVE omega_cross_matrix * omega_cross_matrix is right for omega_cross_matrix^2
	# http://ethaneade.com/lie.pdf pg 10
		A = np.sin(theta) / theta
		B = (1 - np.cos(theta))/(theta ** 2)
		C = (1 - A) / (theta ** 2)
		D = (theta - np.sin(theta)) / (theta ** 3)

		V = np.eye(3) + B * omega_cross_matrix + C * np.dot(omega_cross_matrix, omega_cross_matrix)
		R = np.eye(3) + A * omega_cross_matrix + B * np.dot(omega_cross_matrix, omega_cross_matrix)

	V_translation = np.matmul(V, translation_vector)

	return np.array([
		[R[0, 0], R[0, 1], R[0, 2], V_translation[0, 0]],
		[R[1, 0], R[1, 1], R[1, 2], V_translation[1, 0]],
		[R[2, 0], R[2, 1], R[2, 2], V_translation[2, 0]],
		[0, 0, 0, 1]
	])


def residuals(previous_image, current_image, xi, inverse_depth):
	# Assuming that previous_image points are in pixel coordinates, not sure IF THAT IS CORRECT!!!
	# Apply se(3): http://ethaneade.com/lie_groups.pdf
	exp_se_3 = expSe3Mapping(xi)

	# print("residuals() previous_image.shape", previous_image.shape)

	residuals = np.zeros((previous_image.shape[0] * previous_image.shape[1], 1))

	for i in range(previous_image.shape[0]):
		for j in range(previous_image.shape[1]):
			i_d = inverse_depth[i, j]

			vec = np.array([
				[j / i_d],
				[i / i_d],
				[1 / i_d],
				[1]
			])

			# print(exp_se_3)
			# print(vec)
			unprojected_point = np.matmul(exp_se_3, vec)
			image_projected_point = projectPoint(unprojected_point)

			# So this projected point is the estimated position of the point in the second image (the one transformed by xi)
			# Call the second image curr
			# Call first prev
			# The residual (https://github.com/krrish94/dvo_python/blob/master/photometric_alignment.py)
			# seems to be the difference in intensity btwn the previous image at x, y and the current image at the warped points
			# So here I got the image_projected_point, which is in the current image
			# So I think the residual is then the difference in intensity of the original point and image_projected_point in the current image???

			# print(np.round(image_projected_point[0, 0]))
			# print(j * previous_image.shape[0] + i)
			# print(image_projected_point)

			# Points which are projected on the wrong side of the camera shouldn't count
			# print("residuals() image_projected_point", image_projected_point)
			if (image_projected_point[2] > 0) and (image_projected_point[1] < previous_image.shape[0]) and (image_projected_point[0] < previous_image.shape[1]):
				intensity_current = bilinear_interpolation(current_image, image_projected_point[1][0], image_projected_point[0][0], previous_image.shape[1], previous_image.shape[0])
				residuals[j * previous_image.shape[0] + i][0] = intensity_current - previous_image[i, j]

	return residuals


def jacobian(current_image, previous_image, K, K_inv, xi, inverse_depth, residuals_stacked):
	fx = K[0, 0]
	fy = K[1, 1]

	grad_current_x = cv2.Sobel(current_image, cv2.CV_64F, 1, 0, ksize=3)
	grad_current_y = cv2.Sobel(current_image, cv2.CV_64F, 0, 1, ksize=3)

	jacobian = np.zeros((current_image.shape[0] * current_image.shape[1], 6))

	# print("jacobian() xi", xi)
	SE_3 = expSe3Mapping(xi)
	rotation = SE_3[0:3, 0:3]
	translation = SE_3[0:3, 3]
	# K_inv = np.linalg.inv(K)

	rotation_K_inv = np.matmul(rotation, K_inv)
	# print("jacobian() rotation", rotation)
	# print("jacobian() check rotation is rotation", np.matmul(rotation.T, rotation), np.matmul(rotation, rotation.T))
	# print("jacobian() translation", translation)
	# print("jacobian() K_inv", K_inv)

	for y in range(current_image.shape[0]):
		for x in range(current_image.shape[1]):
			depth = 1 / inverse_depth[y, x]
			point = np.array([depth * x, depth * y, depth]).reshape(3, 1)

			unprojected_point = np.matmul(rotation_K_inv, point) + translation.reshape(3, 1)

			# print("jacobian() rotation_K_inv", rotation_K_inv)
			# print("jacobian() translation", translation)
			# print("jacobian() point", point)
			# print("jacobian() unprojected_point", unprojected_point)

			unprojected_x = unprojected_point[0]
			unprojected_y = unprojected_point[1]
			unprojected_z = unprojected_point[2]

			grad_focal = np.array([[grad_current_x[y, x] * fx, grad_current_y[y, x] * fy]])
			intermittent = np.array([
				[1, 0, -unprojected_x/unprojected_z, -(unprojected_x * unprojected_y)/unprojected_z, unprojected_z + ((unprojected_x**2)/unprojected_z), -unprojected_y],
				[0, 1, -unprojected_y/unprojected_z, -(unprojected_z + ((unprojected_y**2)/unprojected_z)), (unprojected_x * unprojected_y)/unprojected_z, unprojected_x]
			])

			jacobian[x * current_image.shape[0] + y, :] = (1 / unprojected_z) * np.matmul(grad_focal, intermittent)

	# residuals_stacked = residuals(previous_image, current_image, xi, inverse_depth)

	# d_xi = -np.inv(jacobian.T * jacobian) * jacobian.T * residuals_stacked

	return jacobian


def adjust_xi(delta_xi, current_xi):
	return poseConcatenation(delta_xi, current_xi)


# NOTE:!!!!!!! THE INVERSE DEPTH HERE IS OF PREVIOUS IMAGE
def getDrDD(previous_inverse_depth, exp_se_3, p_x, p_y):
	x_prime = (exp_se_3[0, 0] * p_x + exp_se_3[0, 1] * p_y + exp_se_3[0, 2]) * (1 / previous_inverse_depth) + exp_se_3[0, 3]
	y_prime = (exp_se_3[1, 0] * p_x + exp_se_3[1, 1] * p_y + exp_se_3[1, 2]) * (1 / previous_inverse_depth) + exp_se_3[1, 3]
	z_prime = (exp_se_3[2, 0] * p_x + exp_se_3[2, 1] * p_y + exp_se_3[2, 2]) * (1 / previous_inverse_depth) + exp_se_3[2, 3]

	dx_prime = -((exp_se_3[0, 0] * p_x + exp_se_3[0, 1] * p_y + exp_se_3[0, 2]))* (1 / (previous_inverse_depth**2))
	dy_prime = -((exp_se_3[1, 0] * p_x + exp_se_3[1, 1] * p_y + exp_se_3[1, 2]))* (1 / (previous_inverse_depth**2))
	dz_prime = -((exp_se_3[2, 0] * p_x + exp_se_3[2, 1] * p_y + exp_se_3[2, 2]))* (1 / (previous_inverse_depth**2))

	x_z = (z_prime*dx_prime + x_prime * dz_prime) / (z_prime**2)
	y_z = (z_prime*dy_prime + y_prime * dz_prime) / (z_prime**2)
	one_z = dz_prime / (z_prime**2) 		# Using quotient rule here too

	omega = np.array([
		[x_z],
		[y_z],
		[one_z]
	])


def variance(image_intensities, inverse_depth_variance, xi):
	# Variance of image_intensity? idk but I guess

	# exp_se_3 = expSe3Mapping(xi)

	# image_intensity_variance = np.var(image_intensities)

	# _variance = np.zeros(image_intensities.shape[0] * image_intensities.shape[1], 1)

	# for i in image_intensities.shape[0]:
	# 	for j in image_intensities.shape[1]:


	# 		_variance[j * image_intensities.shape[0] + i, 1] = 2 * image_intensity_variance + inverse_depth_variance[i, j]

	# estimate the variance with propagation of uncertainty?
	# So I guess to do that I'll need to get J_residual * covariance_whatever_residual_is_function_of * J_residual...right?

	# covariance_whatever_residual_is_function_of....
	pass


def calculateError(stacked_residuals, photometric_variance):
	# TODO: Use Huber Norm as error. It's robust to noise. In the mean time, this will probably do.
	# residual_squared = np.dot(stacked_residuals.T, stacked_residuals)

	return np.dot(stacked_residuals.T, stacked_residuals)


def warpImage(current_image, previous_image, K, inverse_depth, keyframe):

	# I think this makes sense as initial guess of xi (identity transformation)
	# xi = np.array([
	# 	[2],
	# 	[1e-50],
	# 	[2],
	# 	[2],
	# 	[1e-50],
	# 	[1e-50]
	# ])

	# print("warpImage() xi to matrix test\n", expSe3Mapping(xi))

	# identity = np.array([
	# 	[0, 0, 0, 1],
	# 	[0, 0, 0, 0],
	# 	[0, 0, 0, 0],
	# 	[0, 0, 0, 0]
	# ])
	# xi = logSe3Mapping(identity)
	# print("warpImage() test xi", xi)
	xi = np.array([
		[1e-50],
		[1e-50],
		[1e-50],
		[1e-50],
		[1e-50],
		[1e-50]
	])


	prev_xi = xi

	# xi = np.random.normal(0, 1, (6, 1))

	# threshold = 0.9999999999999999999999999999999 	# This is kind of randomly chosen, fix later
	threshold = 1 - 1e-5
	threshold = 0.9995
	# threshold = 0.7

	# stacked_residuals = residuals(previous_image, current_image, xi, inverse_depth)
	residuals, cache_point3d = externalComputeResiduals(previous_image, inverse_depth, current_image, K, xi)

	# print("warpImage() stacked_residuals", stacked_residuals)
	# print("warpImage() stacked_residuals.shape", stacked_residuals.shape)
	K_inv = np.linalg.inv(K)
	# J = jacobian(current_image, previous_image, K, K_inv, xi, inverse_depth, stacked_residuals)
	J = externalComputeJacobian(previous_image, inverse_depth, current_image, K, xi, residuals, cache_point3d)
	photometric_variance = getPhotometricVariance(keyframe, J)
	stacked_residuals = np.zeros((residuals.shape[0] * residuals.shape[1], 1))
	stacked_J = np.zeros((J.shape[0] * J.shape[1], 6))
	# print(stacked_residuals.shape)

	for i in range(residuals.shape[0]):
		for j in range(residuals.shape[1]):
			stacked_J[j * current_image.shape[0] + i, :] = J[i, j, :]
			stacked_residuals[j * current_image.shape[0] + i, 0] = residuals[i, j]

	error = calculateError(stacked_residuals, photometric_variance)
	prev_error = 2*error
	# print("warpImage() error", error)
	d_xi = -np.matmul(np.linalg.inv(np.matmul(stacked_J.T, stacked_J)), np.matmul(stacked_J.T, stacked_residuals))
	prev_xi = xi
	xi = adjust_xi(d_xi, xi)

	while ((error/prev_error) < threshold):
		J = externalComputeJacobian(previous_image, inverse_depth, current_image, K, xi, residuals, cache_point3d)
		# stacked_J = np.zeros((J.shape[0] * J.shape[1], 6))
		# stacked_residuals = np.zeros((residuals.shape[0] * residuals.shape[1], 1))
		residuals, cache_point3d = externalComputeResiduals(previous_image, inverse_depth, current_image, K, xi)

		for i in range(J.shape[0]):
			for j in range(J.shape[1]):
				stacked_J[j * current_image.shape[0] + i, :] = J[i, j, :]
				stacked_residuals[j * current_image.shape[0] + i, 0] = residuals[i, j]

		d_xi = -np.matmul(np.linalg.inv(np.matmul(stacked_J.T, stacked_J)), np.matmul(stacked_J.T, stacked_residuals))
		prev_xi = xi
		xi = adjust_xi(d_xi, xi)
		# J = jacobian(current_image, previous_image, K, K_inv, xi, inverse_depth, stacked_residuals)

		prev_error = error
		photometric_variance = getPhotometricVariance(keyframe, J)
		
		error = calculateError(stacked_residuals, photometric_variance)
		print("warpImage() previous error", prev_error)
		print("warpImage() error", error)
		print("warpImage() ratio of error", error/prev_error)


	if (error - prev_error) > 0: 
		return prev_xi

	return xi


def main():
	images, K = readFrames(sys.argv[1], sys.argv[2])
	previous_image = images[0]
	current_image = images[1]

	inverse_depth = np.random.normal(4, 1, previous_image.shape)
	
	xi = warpImage(current_image, previous_image, K, inverse_depth)

	print("Final rotation and translation", expSe3Mapping(xi))


if __name__ == "__main__":
	main()



