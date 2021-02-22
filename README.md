# ROSBA - Robust Sparse Bundle Adjustment


This C++ repository implements a number of state-of-the-art algorithms for **large-scale Bundle Adjustment**, including 
- IRLS: Iteratively Re-weighted Least Squares
- M-HQ: Multiplicative Half Quadratic Lifting
- GOM+: Graduated Non-convexity.
- MOO:  Robust fitting with Multi-Objective Optimization (Zach 2019)
- ASKER: Adaptive Kernel Scaling
- ReGeMM: Relaxed Generalized Majorization Minimization (ReGeMM)

See the following [paper](https://www.researchgate.net/publication/349494643_Escaping_Poor_Local_Minima_in_Large_Scale_Robust_Estimation_Generalized_Majorization-Minimization_and_Filter_Methods) for more details as well as the comparisons between the methods.

## Usage
The methods are written is C++, developed mainly based on the [SSBA library](https://github.com/chzach/SSBA). In this release, we use Conjugate Gradient to solver the linear system. Therefore, we can handle larger datasets compared to previous SSBA releases.

A CMake file is provided. Before compilation, please set OPTIMIZER variable (in the CMakeLists.txt file) to the desired method. For instance, use 

``` set (OPTIMIZER schur_irls) ```
to compile the IRLS algorithm. The output executable will be `bundle_irls`

Similarly, set the ``OPTIMIZER`` variable to the following to compile the associated algorithms:
- schur_lifting: M-HQ
- schur_gnc: GOM+
- schur_gnc_moo: MOO
- schur_filter: ASKER
- schur_gemm: ReGeMM

After setting the desired method in the CMakeLists.txt file, create a build folder (from the source directory) and start compiling:

``` 
mkdir build
cd build
cmake ..
make -j8
```


# Testing 
A sample input file is provided in the Dataset folder. After compiling, run

```
bundle_<method> ../Dataset/ladybug-49.txt
```

# Issues
Please create issue on this repository, or contact ```myemail@chalmers.se``` , where 
```
myemail=huul
```



