import pptk
import numpy as np
from plyfile import PlyData
import sys

def main():
	with open(sys.argv[1], 'rb') as f:
		plydata = PlyData.read(f)
		print(plydata.elements[0].data.shape)

		points = plydata.elements[0].data
		points_to_display = []
		for point in points:
			points_to_display.append([point[0], point[1], point[2]])

		v = pptk.viewer(np.array(points_to_display))
		v.wait()
		v.close()


if __name__ == "__main__":
	main()
