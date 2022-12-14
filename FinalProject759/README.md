# Monte Carlo Simulations in Option Pricing


#### How to clone repo
```shell
git clone git@git.doit.wisc.edu:HSAGAR2/repo759.git
cd repo759
cd FinalProject759
```

### Steps to run CPU code
#### 1. Install lapack

```shell
mkdir -p external
cd external
wget https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.11.tar.gz
tar -xvf v3.11.tar.gz
cd lapack-3.11/
cp make.inc.example make.inc  # use example make as make
make
```

#### 2. Install blas

```shell
mkdir -p external
cd external
wget http://www.netlib.org/blas/blas-3.11.0.tgz
tar -xvf blas-3.11.0.tgz  # unzip the blas source files
cd BLAS-3.11.0/ 
make
mv blas_LINUX.a libblas.a
```

#### 3. Run cpu_run.sh
```shell
bash cpu_run.sh
```
### Steps to run GPU code
#### 1. Run gpu_run.sh
```shell
bash gpu_run.sh
```

## Combined shell scripts to run cpu and gpu code
```shell
#!/usr/bin/env zsh

git clone git@git.doit.wisc.edu:HSAGAR2/repo759.git
cd repo759
cd FinalProject759

mkdir -p external
cd external
wget https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.11.tar.gz
tar -xvf v3.11.tar.gz
cd lapack-3.11/
cp make.inc.example make.inc  # use example make as make
make
cd ../../

mkdir -p external
cd external
wget http://www.netlib.org/blas/blas-3.11.0.tgz
tar -xvf blas-3.11.0.tgz  # unzip the blas source files
cd BLAS-3.11.0/ 
make
mv blas_LINUX.a libblas.a
cd ../../


euler-mini -G
rm mc_gpu
rm mc_cpu

nvcc mc_cpu.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o mc_cpu -llapack -lblas -lm -lcurand -lgfortran -Xcompiler -Lexternal/lapack-3.11  -Xcompiler -Lexternal/BLAS-3.11.0 
nvcc mc_gpu.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o mc_gpu -lcurand 

./mc_cpu
./mc_gpu
```

