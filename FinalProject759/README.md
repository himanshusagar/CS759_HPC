# Monte Carlo Simulations in Option Pricing


## How to clone repo
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
#### 1. Run compiler.sh
```shell
bash compiler.sh
```

#### 2. Run runner.sh
```shell
bash runner.sh
```