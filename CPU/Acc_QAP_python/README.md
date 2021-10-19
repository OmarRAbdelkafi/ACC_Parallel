# CPU - python
For this acceleration we use the well known QAP.
For the python acceleration we use the numba library.
We just accelerate the objectif function.  

We propose three mode :
-Normal:  Normal CPU implementation runing on the python SDK
-jit: the just in time execution who transform the python code on machine
-parallel: Parallel CPU multiprocessor. It is efficient when we have multiple powerful processor
