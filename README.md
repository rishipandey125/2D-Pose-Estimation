# 2D Pose Estimation Project
Pose estimation project implementing a convolution neural network constructed from the MPII human pose dataset to estimate human joints based off of video footage. I developed this program to gain a deeper understanding of how computer vision and machine learning can be levaraged to identify human poses to create a mobile motion capture pipeline for film, animation, and immersive development. 

# Process and Implementation


# Future Improvements
To leverage computer vision pose estimation for motion capture, two key features must be added to this project. The first would be 3D reconstruction. Using a mathematical method 3D pose data must be reconstructed to transfer over to motion capture/animation data. With the inclusion of depth sensors on future mobile devices this process may be easier in the future, but for the time being mathematical reconstruction can be explored. 

From scavenging the web, this paper explained the process of 3D reconstruction of 2D points best for me. 
https://www.researchgate.net/publication/258220500_Reconstructing_The_Missing_Dimension_From_2D_To_3D_Human_Pose_Estimation

I am currently working on implementing its explanation into this project. 

The second key improvement would be writing 3D reconstructed pose data to FBX animation files, to complete the mobile performance capture pipeline. My plan after 3D reconstruction, is to use the FBX SDK provided by Autodesk to write the data to animation files. This process would also include tweaking the data to correctly fit a characters animation rig. 
