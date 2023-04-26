# van-karman-vertex-sheet

Simulation of Flow around a cylinder using lattice boltzman method

![An example of an iteration step](./assets/step.png)

**To Start the simulation run**

`python main.py`

It will create snapshots of images at various iterations in the images folder

**To Create a video simulation using the images generated in image folder using ffmpeg**

`ffmpeg -framerate 30 -i %d.png output.mp4`
