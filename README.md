# freeze-h5
Freeze-h5 is an util that converts the hdf5 model to pb or pbtxt.

The all weights of the converted model is frozen.

This script run cv2.readNet() as a way to verify that the model has been converted well.

If not converted normally, cv2.readNet() will not run or return empty net.

All you have to do is place the model file to convert and run the script.

```bash
$ cd freeze-h5
$ mv my_model.h5 model.h5
$ python freeze-h5.py
```
