class GlobalMap:
	keyframes = []

	def __init__(self, keyframe):
		self.keyframes.append(keyframe)


	def addKeyframe(self, keyframe):
		self.keyframes.append(keyframe)


	def getKeyframes(self, keyframe):
		return self.keyframes