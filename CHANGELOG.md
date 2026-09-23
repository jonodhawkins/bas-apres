# Changelog

## [v0.4.2] - 2025-04-18

### Added

- Add support for rewriting (subsetting) and plotting multi-attenuator data

## [v0.4.1] - 2025-04-17

### Fixed

- Fix a bug whereby the dimension lengths are in the correct order but the corresponding dimension keys are not

## [v0.4.0] - 2025-04-17

### Added

- Add GH Actions workflow for building API docs

### Changed

- Update the package documentation link to point to the new gh-pages API docs

### Fixed

- Fix bug in reshaping when more than one attenuator setting used in a burst

## [v0.3.0] - 2025-02-24

### Added

- Add docs on loading data from remote cloud storage
- Add support for opening remote datasets to CLI scripts - options for connecting to remote services
- Add support for opening remote datasets

### Fixed

- Fix support for installing remote extra

### Changed

- Update authors - contribution for adding support for remote loading of data
- Support for opening files in binary mode - this is now the default

## [v0.2.0] - 2024-10-15

### Added

- Add support for optional, additional dimensions
- Add support to cope with a variable number of dimensions
- Add support for averaged data (Average=1)
- Add details of how to install the package to the README
- Add details of the repository to pyproject.toml
- Add support for subsetting bursts, in addition to subbursts and samples
- Add console script entry points to pyproject.toml
- Add end-to-end tests for the conversion of ApRES -> netCDF -> ApRES
- Add useful test script to drive two-way conversion process between ApRES and netCDF
- Add nc_to_apres.py to the list of scripts installed with the package
- Add support for plotting radargrams
- Add support for a 'forgiving' mode when reading data
- Add support for averaged/stacked data files
- Add function to reset DEFAULTS parsing tokens to initial settings
- Add checks to determine file format version and parse accordingly

### Changed

- Update documentation to reflect support for multi-attenuator data
- Update plot_apres to handle multi-attenuator data
- Improve the handling of the flatten option
- A burst should store its dimension keys, rather than rely on the parser constants.  This enables support for a variable number of dimensions
- Minor mod.  The RE for detecting the end of header sometimes raises a deprecation warning about invalid escape sequence in pytest.  This should be a raw string
- Minor mods. to rationalise suffix handling for ApRES raw and converted netCDF
- Make consistent the subsetting nomenclature for subbursts
- Update the usage docs (help, README) to reflect that the scripts are now package modules
- Update end-to-end test script to use module invocation for scripts
- Update all formatted strings to be f-strings
- Support for ApRES timeseries files.  Largely what was ApRESFile is now ApRESBurst, and the context manager and netCDF conversion functions are now in an ApRESFile object that wraps a number of ApRESBurst objects
- Only read the required number of items into the data array, determined by the data_type and the product of the data_shape
- Allow for a converted netCDF file to be converted back to the original ApRES file
- Make path optional in constructor and open()
- Ensure default parsing tokens are first in the alternatives list
- Ensure exceptions are not suppressed when using as a context manager
- Strip leading/trailing whitespace from header key/value pairs
- Make suffix-based file type check case-insensitive
