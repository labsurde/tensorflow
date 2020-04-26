

- Converting pbtxt
    - See https://colab.research.google.com/drive/1gs9xLfsyKe17DchtqiKT9f_ROSxamjLi?authuser=1#scrollTo=4UPMrylPhsqm for converting script

- Converting pb to tflite
    - `$ docker run -v /home/eric:/home/eric -it -w $(pwd) tensorflow/tensorflow:nightly python3 convert.py`

- Build
    - `$ bazel build -c opt //tensorflow/lite/examples/unknown_dim:unknown_dim`

