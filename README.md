# Complex Wrinkle Field Decomposition
This repository is the follow up work of the paper: "Complex Wrinkle Field Evolution".

## To download
```
git clone https://github.com/zhenchen-jay/Complex-Wrinkle-Field.git 
```

## Dependencies
- [Libigl](https://github.com/libigl/libigl.git)
- [Polyscope](https://github.com/nmwsharp/polyscope.git)
- [Geometry-Central](https://github.com/nmwsharp/geometry-central.git) 
- [TBB](https://github.com/wjakob/tbb.git)
- [Spectra](https://github.com/yixuan/spectra.git)
- [Suite Sparse](https://people.engr.tamu.edu/davis/suitesparse.html)

All the dependencies are solved by Fetcontent, except Suite Sparse and Spectra (see below for instructions for these two libraries). 

## build with spectra
In order to build with Spectra with the same Eigen version of libigl, please comment out line 24-26 of the /build/_deps/spectra-src/CMakeLists.txt:
```
# find_package(Eigen3 NO_MODULE REQUIRED)
# set_package_properties(Eigen3 PROPERTIES TYPE REQUIRED PURPOSE "C++ vector data structures")
# message(STATUS "Found Eigen3 Version: ${Eigen3_VERSION} Path: ${Eigen3_DIR}")
```

## build with Suite-Sparse
This part is tricky, for linux, you should use 
```
sudo apt-get update -y
sudo apt-get install -y libsuitesparse-dev
```

For macOS, this can be done with [Homebrew](https://brew.sh/):
```
brew install suite-sparse
```

For windows, please follow the guidence provided in [suitesparse-metis-for-windows](https://github.com/jlblancoc/suitesparse-metis-for-windows).


## Build and Run
```
mkdir build
cd build
cmake ..
make -j4
```
It contains several executable files. 
- `CWFUpsamplingGui_bin` is the one used to visualize the upsampling results
- `CWFDecompositionGui_bin` is the decomposition algorithm under development
To run them, you can try
```
./CWFUpsamplingGui_bin -i data/bunny/data.json
```
or
```
./CWFDecompositionGui_bin -i data/bunny/data.json
```


## Issues
When compiling on MacOS with C++17, you may encounter this issue: 
```
build/_deps/comiso-src/ext/gmm-4.2/include/gmm/gmm_domain_decomp.h:84:2: error: ISO C++17 does not allow 'register' storage class specifier [-Wregister]
```
To solve this, please replace `register double` by `double`.  
