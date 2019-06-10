# Raven Pillmann
import sys
import cv2
import glob
import numpy as np


def extractFramesFromVideo(video):
	vid = cv2.VideoCapture(video)

	frames = []
	success = True

	while (success):
		success, frame = vid.read()
		
		if success:
			grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frames.append(grayscaled_frame)

	vid.release()

	return frames


def readFrames(image_directory_path, intrinsic_camera_parameters_path):
	# Read and downsample frames (gaussian blur then downsample)
	images = [cv2.pyrDown(cv2.imread(file, cv2.COLOR_BGR2GRAY)) for file in sorted(glob.glob(image_directory_path + "/*.png"))]


	with open(intrinsic_camera_parameters_path) as f:
		content = f.readlines()
		# you may also want to remove whitespace characters like `\n` at the end of each line
		content = [x.strip() for x in content]

		first_line_split = content[0].split()
		second_line_split = content[1].split()

		# I believe these are right, given https://github.com/tum-vision/lsd_slam/blob/master/README.md#313-camera-calibration
		fx_width = float(first_line_split[0])
		fy_height = float(first_line_split[1])
		cx_width = float(first_line_split[2])
		cy_height = float(first_line_split[3])

		width = int(second_line_split[0])
		height = int(second_line_split[1])

	K = np.array([
		[np.round_(fx_width * width), 0, np.round_(cx_width * width)],
		[0, np.round_(fy_height * height), np.round_(cy_height * height)],
		[0, 0, 1]
	])

	# dimensions = (height, width)

	return images, K


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Please include a video to read")
	else:
		images, K, dimensions = readFrames(sys.argv[1], sys.argv[2])
		print(len(images))
		print(K)
	# else:
	# 	frames = extractFramesFromVideo(sys.argv[1])
	# 	print("Frames extracted from video")
	# 	print(len(frames))
		# cv2.imshow('First Frame', frames[len(frames)-1])
		# if cv2.waitKey(1) & 0xFF == ord('q'):
		# 	cv2.destroyAllWindows()