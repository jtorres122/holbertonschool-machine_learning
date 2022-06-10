<h1 align="center">0x04. Convolutions and Pooling</h1>

### Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

### General

- What is a convolution?
  - Convolution is the mathematical operation on two functions, f and g, that produces a third function(f \* g),\
    that expresses how the shape of one is modified by the other.
- What is max pooling? average pooling?
  - Max pooling
- What is a kernel/filter?
  - A kernel, also known as a filter, convolution matrix or mask, is a small matrix for blurring,\
    sharpening, embossing, edge detection and more. It's accomplished by doing a convolution between\
    the kernel and an image.
- What is padding?
  -
- What is “same” padding? “valid” padding?
- What is a stride?
- What are channels?
- How to perform a convolution over an image
- How to perform max/average pooling over an image

### Requirements

#### General

- Allowed editors: vi, vim, emacs
- All your files will be interpreted/compiled on Ubuntu 20.04 LTS using python3 (version 3.8)
- Your files will be executed with numpy (version 1.19.2)
- All your files should end with a new line
- The first line of all your files should be exactly #!/usr/bin/env python3
- A README.md file, at the root of the folder of the project, is mandatory
- Your code should use the pycodestyle style (version 2.6)
- All your modules should have documentation (python3 -c 'print(**import**("my_module").**doc**)')
- All your classes should have documentation (python3 -c 'print(**import**("my_module").MyClass.**doc**)')
- All your functions (inside and outside a class) should have documentation (python3 -c 'print(**import**("my_module").my_function.**doc**)' and python3 -c 'print(**import**("my_module").MyClass.my_function.**doc**)')
- Unless otherwise noted, you are not allowed to import any module except import numpy as np
- You are not allowed to use np.convolve
- All your files must be executable
- The length of your files will be tested using wc
