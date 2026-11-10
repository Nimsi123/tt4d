# Table tennis calibration

Processing a video will output:
- a yaml file with the camera calibration file called "calib.json" following this format if the camera is static:
```yaml
tvec: [3, 6.7, 19,0]
rvec: [0.23, 1.4, -0.9]
f: 1500
width: 1280
height: 720
```
- a flag "done.txt" if the calibration was successful
- a flag "failed.txt" if the calibration failed with the reason inside the file
- a flag "moves.txt" if the camera is not static. I will contain an estimate of the camera movement as the integral of the translations and rotations
- an image called "calibrated.png" of the video with the projected table axis and detected table corners
- a file called "calib.csv" containing the calibration for everyframe in the video. It will have the following header: [tvec\_x, tvec\_y, tvec\_z, rvec\_x, rvec\_y, rvec\_z, f]. If the calibration failed, all the entries will be equal to -1.
- an plot of the estimated calibration over all the video as an image: estimates.png
