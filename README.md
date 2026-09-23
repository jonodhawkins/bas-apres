# apres

This project enables ApRES .dat files to be read, rewritten, and converted.

## The apres package

The apres package contains a number of classes for working with ApRES .dat files.

* The `ApRESBurst` class manages reading/writing a single burst block (the textual header and the binary data) from/to an opened ApRES .dat file.
* The `ApRESFile` class is a context manager for ApRES .dat files.  The class wraps the `ApRESBurst` class and can read a variable number of burst blocks from a file (i.e. a single burst file or a timeseries file).  It also contains methods for subsetting an ApRES file, and for converting the file to netCDF4.

### Loading data from remote cloud storage

As of version 0.3.0, the apres package has the ability to load data held on remote cloud storage platforms such as those offered by Google or Amazon.  To take advantage of this, the optional `remote` package extra must be installed as well as the specific [fsspec](https://filesystem-spec.readthedocs.io/en/stable/index.html) package for the required cloud storage platforms (e.g. `gcsfs` or `s3fs`).  See [Install remote extra](#install-remote-extra) for installation details.

To specify a remote ApRES file, the `path` passed to `ApRESFile()` should be a valid URI for the given cloud storage platform, e.g. `s3://<bucket_name>/<object_path>`.  In addition, any filesystem-specific `fsspec` options required for accessing the file (e.g. credentials) should be specified in the `fs_opts` `dict`, passed as a *kwarg* to `ApRESFile()`.  For example:

```python
    with ApRESFile('s3://bucket/data.dat', fs_opts={'anon': True}) as f:
        for burst in f.read():
            print(burst.header)
```

## Install

The package can be installed from PyPI (note the package distribution name):

```bash
$ pip install bas-apres
```

### Install remote extra

To load data from remote cloud storage platforms, the `remote` extra must be installed:

```bash
$ pip install bas-apres[remote]
```

In addition, specific `fsspec` packages for each cloud storage platform to be used must also be installed.  For example, to load data from Amazon S3 buckets:

```bash
$ pip install s3fs
```

## Simple utility scripts

The package contains a number of scripts for converting from/to netCDF, plotting the data etc.  When installing the package (e.g., via `pip install`) then these scripts are available as commands and added to the `PATH`.  So for instance, you can just call `apres_to_nc`.  If running the scripts from a `clone` of the source code repository, then they need to be run as modules, e.g. `python -m apres.apres_to_nc`.

### Input files held on remote cloud storage

The utility scripts have the ability to load data from remote cloud storage.  To do so, the input filename must be specified as a valid URI for the particular cloud storage platform.  In addtion, any `fsspec` options required for accessing the remote resource should be specified as a JSON string (via the `-o` option) or as a JSON file (via the `-O` option).  The latter is a better choice if the `fsspec` options include sensitive credentials.  For example, to anonymously access publicly open data from an S3 bucket:

```bash
$ python -m apres.read_apres -o '{"anon": true}' s3://apres-tests/short-test-data.dat
```

or equivalently, given a JSON file:

```bash
$ cat fs_opts.json 
{"anon": true}
$ python -m apres.read_apres -O fs_opts.json s3://apres-tests/short-test-data.dat
```

### apres_to_nc.py

This script converts an ApRES .dat file to netCDF4.  The default output netCDF filename has the same name as the input .dat file, but with a .nc file suffix.  Optionally an alternative output netCDF filename can be explicitly given.

The conversion is a straightforward mapping, in that each burst becomes a netCDF4 group, where the header lines become group attributes, and the data are stored as the group data.  How the data are stored depends on the number of attenuators used when acquiring the data:

* Single-attenuator configuration: The data are stored as a `NSubBursts` x `N_ADC_SAMPLES` 2D data array.
* Multi-attenuator configuration: The data are stored as a `NSubBursts` x `nAttenuators` x `N_ADC_SAMPLES` 3D data array.

### nc_to_apres.py

This script converts a previously converted netCDF4 file back to the original ApRES .dat file.  The default output ApRES .dat filename has the same name as the input netCDF file, but with a .dat file suffix.  Optionally an alternative output ApRES filename can be explicitly given.

The conversion is a straightforward reversal of the original conversion.  For the newer ApRES .dat file format version, this should be identical to the original file.  For the older ApRES .dat file format version, there will likely be small differences in the whitespace in the textual header.  Ignoring these insignificant changes in whitespace (e.g. `diff -awB orig.dat reconstructed.dat`), the files will be identical.

### plot_apres.py

This script will plot the `N_ADC_SAMPLES` vs `NSubBursts` as a radargram, for each burst in the file (or group for converted netCDF files).  If the `Average` header item > 0 (and so each subburst has been aggregated), then the script will instead plot the first subburst as a single trace.  If the `nAttenuators` header item > 1, then each attenuator's data are plotted separately.  The file can be either an ApRES .dat file, or a converted netCDF file.

### read_apres.py

This script will read the given ApRES .dat file, and for each burst, print the results of parsing the header, such as the dimensions of the data array, and the parsed header dictionary.  It will also *head* the data section (by default the first 10 samples of the first 10 subbursts), to give an initial simple look at the data.  If the data were acquired using multiple attenuators, then the number of samples shown will be multiplied by the number of attenuators.

The script's primary purpose is as a simple example of how to use the `ApRESFile` class to read an ApRES .dat file.

### write_apres.py

This script will read the given input ApRES .dat file, and for each burst, write the header and data to the given output ApRES .dat file.  Optionally a subset of bursts can be written out, specified as the first `bursts` bursts of the input file.  In addition a subset of each burst can be written out, specified as the first `subbursts` subbursts, and the first `samples` ADC samples of these subbursts.  If `bursts`, `subbursts` and `samples` are not specified, then the output file is identical to the input file.

The script's primary purpose is as a simple example of how to use the `ApRESFile` class to rewrite an ApRES .dat file.

## Command line usage

### apres_to_nc.py

```python
usage: apres_to_nc [-h] [-o FS_OPTS | -O FS_OPTS] [-V] infile [outfile]

convert a raw ApRES file to a netCDF4 file

positional arguments:
  infile                ApRES raw file
  outfile               converted netCDF file

optional arguments:
  -h, --help            show this help message and exit
  -o FS_OPTS, --fs-opts FS_OPTS
                        fsspec filesystem options from a JSON string
  -O FS_OPTS, --fs-opts-file FS_OPTS
                        fsspec filesystem options from a JSON file
  -V, --version         show program's version number and exit
```

### nc_to_apres.py

```python
usage: nc_to_apres [-h] [-V] infile [outfile]

recover the raw ApRES file from a converted netCDF4 file

positional arguments:
  infile         converted netCDF file of raw ApRES data
  outfile        ApRES raw file

optional arguments:
  -h, --help     show this help message and exit
  -V, --version  show program's version number and exit
```

### plot_apres.py

```python
usage: plot_apres [-h] [-r | -t] [-g GRID GRID] [-c CONTRAST] [-m CMAP]
                  [-o FS_OPTS | -O FS_OPTS] [-V]
                  filename

plot ApRES data, either from a .dat file, or a converted netCDF file

positional arguments:
  filename              filename

optional arguments:
  -h, --help            show this help message and exit
  -r, --radargram       plot a radargram
  -t, --traces          plot individual traces
  -g GRID GRID, --grid GRID GRID
                        plot the first nrows x ncols traces
  -c CONTRAST, --contrast CONTRAST
                        contrast for the radargram
  -m CMAP, --cmap CMAP  colour map for the radargram
  -o FS_OPTS, --fs-opts FS_OPTS
                        fsspec filesystem options from a JSON string
  -O FS_OPTS, --fs-opts-file FS_OPTS
                        fsspec filesystem options from a JSON file
  -V, --version         show program's version number and exit

Examples

Plot the given ApRES .dat file as a radargram:

python3 -m apres.plot_apres filename.dat

Plot the given converted netCDF file as a radargram:

python3 -m apres.plot_apres filename.nc

Plot the given ApRES file as a radargram, increasing the contrast:

python3 -m apres.plot_apres -c 10 filename.dat

Same as above, but with a blue-white-red colour map:

python3 -m apres.plot_apres -c 10 -m bwr filename.dat

Plot the first trace from the given ApRES file:

python3 -m apres.plot_apres -t filename.dat

Plot the first 6 traces, in a 3x2 grid:

python3 -m apres.plot_apres -t -g 3 2 filename.dat
```

### read_apres.py

```python
usage: read_apres [-h] [-o FS_OPTS | -O FS_OPTS] [-V] infile

print the header, a sample of the data, and diagnostics, from the given raw
ApRES file

positional arguments:
  infile                ApRES raw file

optional arguments:
  -h, --help            show this help message and exit
  -o FS_OPTS, --fs-opts FS_OPTS
                        fsspec filesystem options from a JSON string
  -O FS_OPTS, --fs-opts-file FS_OPTS
                        fsspec filesystem options from a JSON file
  -V, --version         show program's version number and exit
```

### write_apres.py

```python
usage: write_apres [-h] [-b BURSTS] [-u SUBBURSTS] [-s SAMPLES]
                   [-o FS_OPTS | -O FS_OPTS] [-V]
                   infile outfile

read a raw ApRES file and write to another raw ApRES file, optionally
subsetting the data

positional arguments:
  infile                ApRES raw file
  outfile               ApRES raw file

optional arguments:
  -h, --help            show this help message and exit
  -b BURSTS, --bursts BURSTS
                        only write out the first BURSTS bursts
  -u SUBBURSTS, --subbursts SUBBURSTS
                        only write out the first SUBBURSTS subbursts
  -s SAMPLES, --samples SAMPLES
                        only write out the first SAMPLES ADC samples
  -o FS_OPTS, --fs-opts FS_OPTS
                        fsspec filesystem options from a JSON string
  -O FS_OPTS, --fs-opts-file FS_OPTS
                        fsspec filesystem options from a JSON file
  -V, --version         show program's version number and exit
```
