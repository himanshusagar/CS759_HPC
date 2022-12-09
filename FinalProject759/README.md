# Monte Carlo Simulations in Option Pricing

### Formulation

### Steps to build code
#### 1. Install lapack

```shell
mkdir -p external
cd external
wget https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.11.tar.gz
tar -xvf lapack-3.11.0.tar.gz
cd lapack-3.11.0/
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


