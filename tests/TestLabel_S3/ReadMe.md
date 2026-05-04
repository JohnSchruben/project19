# Test for Sprint 3

In this test I did some testing to verify how some of the labels work and to see how well Yolo can indentify objects in different conditions.

## Result

The results show two main take aways:

1) Blurry camera images work but will miss some labeling. It does not recognize the stoplights in image 4

2) Rainy conditions (as in image 5) did not identify anything and will require changes in the future

3) When there is a lot going on (image 3 and 6) Yolo will only label the closest objects and ignore objects far away that appear stacked on top of eachother.