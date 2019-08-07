# Large Scale Direct Monocular SLAM
- Implementation of https://vision.in.tum.de/research/vslam/lsdslam

## TODO
- [X] Tracking: Tracks new camera images, ie estimates their rigid body pose (wrt the current keyframe) Lie-Algebra in se(3), which has six dof. Uses the previous frame as initialization.
- [X] Depth Map Estimation: Either refines or replaces the current keyframe by using tracked frames. Refinement occurs by filtering over small-baseline stereo comparisons between images, and replacement occurs when a camera has moved "too far" from the previous keyframe. In this case, it creates a new keyframe. The new keyframe is initialized by projecting points from existing nearby keyframes into it.
- [ ] Map Optimization: Once a keyframe is done being refined, it's added to a global map. Loop closures and scale drift are detected with a sim(3) Lie-Algebra transformation to close keyframes using direct image alignment.

## Resources
- Lie-Algebras in Computer Vision: https://www.youtube.com/watch?v=khLM8VV8LuM&t=3s


