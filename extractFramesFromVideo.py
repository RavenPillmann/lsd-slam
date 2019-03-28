# Raven Pillmann
import sys
import cv2

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


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Please include a video to read")
	else:
		frames = extractFramesFromVideo(sys.argv[1])
		print("Frames extracted from video")
		print(len(frames))
		# cv2.imshow('First Frame', frames[len(frames)-1])
		# if cv2.waitKey(1) & 0xFF == ord('q'):
		# 	cv2.destroyAllWindows()