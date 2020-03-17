#Team members(4):

  1)Name:Gadiraju Sanjay varma
  
    Email:18pa1a1211@vishnu.edu.in
  2)Name:Abhinav Dayal
  
    Email:abhinav.dayal@vishnu.edu.in
  3)Name:B.Sridevi
  
    Email:sridevi.b@vishnu.edu.in
  4)Name:A.Lakshman Rao
  
    Email:18pa1a0511@vishnu.edu.in

# S9

[Link to assignment](https://github.com/GadirajuSanjayvarma/EVA4/blob/master/S9/EVA04_S9_file1.ipynb)
* We did Albumentation library augmentation and tried cutout, horizontal flip, rgb shift and Rotate along with Normalize and ToTensor
* test transform in only Normalize and ToTensor
* We implemented Gradcam using help from [this library](https://github.com/kazuto1011/grad-cam-pytorch).
* We tested gradcam with both cutout etc. and no transformations other than nortmalize. Results were interesting. In fact on some images it didnt predict well. And we visualize all 4 layer outputs of ResNet. Lots of learning.

# changes to library ([link](https://github.com/GadirajuSanjayvarma/EVA4/tree/master/S9/Eva4-library)

* added gradcam functionality
* added Albumentation transforms

