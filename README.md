# **Overview**

This repository consists of tutorial projects built during the process of learning computer vision, each project should be run through the command line and the respective arguments can be viewed by the **-h argument** in the CMD.

## Files:

1. Ball tracker - Program built to highlight and follow the trail of any **green** colored ball. The ball size and color can be customized according to preference, directly in the code.|
   
2. Coin counter - Program that detects and outlines coins in a given image. It will detect, outline, count and state the number of coins observed in the image.
   
3. Face detection - The main goal is to detect faces and draw bounding boxes around the face. This program can be used as the baseline requirement for further complex tasks such as face recognition
   
4. Face recognition - Program built to detect faces of the **folder-names** and images provided in the dataset folder. The **encoder.py** should be run first to get the embeddings of each of the user. This will be followed by **train_model.py** which will train the actual model using **openface weights initialized**. Finally, we can run the **recognize.py** or **recognize_video.py** to get our respective output in the form of video or image.
   
5. Mobile Scanner - Works to get a **4-point** view of the image and then scan it for better resolution. This uses the **Perspective transformer** code to first get a perspective and then scan the image black-and-white to get a clear visual.
   
6. OMR Scanner - Optimal mark reader, works on grading sheet images to find how many circles were correctly marked. The correct order can be provided in the form of a dictionary in the code. It will show the correctly labeled, incorrectly labeled and the final score.
   
7. Perspective Transformer - Converts an image into a bird's eye point of view. It will find the image corners and transform the image so that the four-points cover the image completely and we get a **top-down** view of the image

Sample images have been provided for each code which can be used to experiment with.