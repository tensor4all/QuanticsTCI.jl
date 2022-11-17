# Setup of `xfacpy`

## With system python

With only one version of python on your system, the setup is relatively straightforward. First, install `PyCall`:
```julia
using Pkg
Pkg.add("PyCall")
```
In this state, `PyCall` will use an internal version of python (supplied by `Conda`). This installation won't be able to find your version of `xfacpy`. Therefore, re-build `PyCall` with your system version of python:
```julia
ENV["PYTHON"] = "... path of the python executable ..."
Pkg.build("PyCall")
```
Re-start Julia, and importing `QuanticsTCI` should now work.

## With multiple versions of python on your system
If you have multiple versions of python on your system, you have to make sure that the version used by `PyCall` is the same version that you compiled your `xfacpy` against. The python you're using has to be compiled in shared mode; if you're using `pyenv`, there are instructions below. Otherwise, you'll have to adapt these (feel free to add your adapted instructions to this document).

## With `pyenv`

If you use `pyenv` on your system, it's possible to specify shared mode during installation of python versions. As an example, we'll use version `3.8.15` in the following. First, install python `3.8.15` like this:
```bash
$ CONFIGURE_OPTS="--enable-shared" pyenv install 3.8.15
```
The environment variable will lead to linking as a shared library. Now, we have to rebuild `PyCall` against the version of python that we just installed. The location of the new `python` executable is something similar to
```julia
ENV["PYTHON"] = "/Users/Ritter.Marc/.pyenv/versions/3.8.15/bin/python"
Pkg.build("PyCall")
```
(Replace `Ritter.Marc` with your username and `3.8.15` with the desired version of python.)

Now, go to your `xfac` directory and compile it using the version we just installed:
```bash
$ cd /path/to/xfac
$ pyenv local 3.8.15
$ pip3 install numpy scipy
$ cmake -S . -D XFAC_BUILD_PYTHON=ON -D CMAKE_BUILD_TYPE=Release -B build
$ cmake --build build --target install
```
We now have a working `xfacpy`, linked against python `3.8.15`. Put the `xfac/python/` folder of `xfac` into `$PYTHONPATH` in Julia, right before loading QuanticsTCI:
```julia
ENV["PYTHONPATH"] = "/path/to/xfac/python/"
using QuanticsTCI
```
