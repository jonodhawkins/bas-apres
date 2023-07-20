###############################################################################
# Project: ApRES
# Purpose: Classes to encapsulate an ApRES file
# Author:  Paul M. Breen
# Date:    2018-09-24
###############################################################################
# Modified:
#   Author:  Jonathan D. Hawkins
#   Date:    2023-07-20
#   Purpose: Parses ApRES header data into processable formats
###############################################################################

__version__ = '0.1.2'

import datetime
import importlib.resources
import json
import os
import re
import sys
import warnings

import numpy as np
from netCDF4 import Dataset

# Load valid *.dat file properties into memory
DAT_FILE_PROPERTIES = json.loads(
    importlib.resources.read_text(
        "apres.resources", 
        "apres_dat_properties.json"
    )
)

class ApRESBurst(object):
    """
    Class for reading and writing ApRES bursts
    """

    DEFAULTS = {
        'autodetect_file_format_version': True,
        'forgive': True,
        'header_markers': {
            'start': '\r\n*** Burst Header ***',
            'end': '\r\n*** End Header ***'
        },
        'end_of_header_re': '\*\*\* End Header',
        'header_line_delim': '=',
        'header_line_eol': '\r\n',
        'data_type_key': 'Average',
        'data_types': ['<u2','<f4','<u4'],
        'data_dim_keys': ['NSubBursts', 'N_ADC_SAMPLES'],
        'data_dim_order': 'C',
        'dds_sys_clock_frequency' : 1e9,
        'dds_registers' : {
            'reg00' : "00000008",
            'reg01' : "000C0820",
            'reg02' : "0D1F41C8",
            'reg0B' : "6666666633333333",
            'reg0C' : "000053E3000053E3",
            'reg0D' : "186A186A",
            'reg0E' : "08B5000000000000",
        }
    }

    ALTERNATIVES = [{
            'header_line_delim': '=',
            'data_dim_keys': ['NSubBursts', 'N_ADC_SAMPLES']
        }, {
            'header_line_delim': ':',
            'data_dim_keys': ['SubBursts in burst', 'Samples']
        }
    ]

    def __init__(self, fp=None):
        """
        Constructor

        * fp is not None: Then header_start will be set to the file's current
        position.  This allows reading through a file one burst block at a
        time.  This is the normal behaviour.
        * fp is None: Allows the burst object to be populated by other means,
        rather than reading from a file

        :param fp: The open file object of the input file
        :type fp: file object
        """

        # Add immutable attribute list to prevent overwrite of header values
        # needs to be first attribute defined
        self._immutable_attr = []

        self.header_start = 0
        self.data_start = -1
        self.header_lines = []
        self.header = {}
        self.data_shape = ()
        self.data_type = '<u2'
        self.data = None

        self.fp = fp

        if self.fp:
            self.header_start = self.fp.tell()

    def reset_init_defaults(self):
        """
        Reset the initial DEFAULTS parsing tokens from the ALTERNATIVES list
        """

        for key in self.ALTERNATIVES[0]:
            self.DEFAULTS[key] = self.ALTERNATIVES[0][key]

    def read_header_lines(self):
        """
        Read the raw header lines from the file

        :returns: The raw header lines
        :rtype: list
        """

        self.data_start = -1
        self.header_lines = []

        self.fp.seek(self.header_start, 0)
        line = self.fp.readline()
        self.header_lines.append(line.rstrip())

        while(line):
            # The data section follows the end of header marker
            if re.match(self.DEFAULTS['end_of_header_re'], line):
                self.data_start = self.fp.tell()
                break

            line = self.fp.readline()
            self.header_lines.append(line.rstrip())

        return self.header_lines
    
    def parse_header_lines(self):
        for line in self.header_lines:
            self._parse_header_line(line)

    def _parse_header_line(self, header_line):
        """
        Parse raw header lines and assign property to the ApRESBurst object

        [more detail here]

        Returns (param_alias, param_value) which correspond to:
            param_alias : machine friendly version of header parameter name
            param_value : parsed version of the header parameter value
        """
        if len(header_line) == 0:
            return None
        
        for header_marker in self.DEFAULTS["header_markers"].values():
            if header_line in header_marker:
                return None

        # Contract header regex based on format delimiter
        header_parts = re.match(r"(.+)" + self.DEFAULTS["header_line_delim"] + r"(.*)", header_line)
        
        if header_parts == None:
            raise ValueError(f"Invalid header line '{header_line}' should match key=value format.")
        
        param_name = header_parts.group(1)
        param_value = header_parts.group(2)

        if not param_name in DAT_FILE_PROPERTIES:
            if self.DEFAULTS['forgive']:
                # TODO: deal with incorrect header here
                return
            else:
                raise KeyError(f"Invalid header key {param_name}.")
            
        param_type = DAT_FILE_PROPERTIES[param_name]["type"]
        if param_type == "int":
            param_value = int(param_value)
        elif param_type =="float":
            param_value = float(param_value)

        # Parse special value if available 
        param_value = ApRESBurst.parse_special_parameter(
            param_name, param_value
        )

        # Get alias name
        param_alias = DAT_FILE_PROPERTIES[param_name]["alias"]

        # assign to object
        self._add_immutable_attribute(param_alias, param_value)

    def _add_immutable_attribute(self, name, value):
        """
        Adds a new immutable attribute to the class
        """
        setattr(self, name, value)
        self._immutable_attr.append(name)
    
    def __setattr__(self, name, value):
        """
        Makes immutable header values
        """
        if name == "_immutable_attr":
            object.__setattr__(self, "_immutable_attr", value)
        elif name in self._immutable_attr:
            raise AttributeError(f"Denied access to header variable {name}.")
        else:
            object.__setattr__(self, name, value)
    
    @staticmethod
    def parse_special_parameter(parameter, value):
        """
        Handle special header parameters (arrays and dates)

        Converts 'Time stamp' header to a datetime.datetime object and 'TxAnt' 
        or 'RxAnt' arrays to Python lists
        """

        # Parse Time Stamp
        if parameter.lower() == "time stamp":
            return datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        
        elif parameter == "TxAnt":
            return [int(v) for v in value.split(",")]

        elif parameter == "RxAnt":
            return [int(v) for v in value.split(",")]#

        else:
            return value
        
    def parse_dds_registers(self):
        """
        Converts DDS registers into meaningful chirp properties

        Converts DDS regsiters into chirp bandwidth and centre frequency
        then assigns additional chirp parameters such as period and sampling
        frequency.
        """
        # Set default register values
        reg0B = self.DEFAULTS["dds_registers"]["reg0B"]
        reg0C = self.DEFAULTS["dds_registers"]["reg0C"]
        reg0D = self.DEFAULTS["dds_registers"]["reg0D"]

        # Check whether we have all of the mandatory registers assigned
        #   if not and we are forgiving, then use default values
        #   if not and we are not forgiving then raise AttributeError
        required_dds_registers = ("Reg0B", "Reg0C", "Reg0D")
        dds_registers_valid = True
        for reg in required_dds_registers:
            if reg not in self._immutable_attr:
                if self.DEFAULTS["forgive"]:
                    dds_registers_valid = False
                    break
                else:
                    raise AttributeError(f"Register {reg} not present in header")
                
        # if they are valid then assign from the values read in from the header
        if dds_registers_valid:
            DDS_REGEX = r'[^0-9A-Fa-f]?([0-9a-fA-F]+)[^0-9a-fA-F]?'

            reg0B = re.match(DDS_REGEX, self.Reg0B).group(1)
            reg0C = re.match(DDS_REGEX, self.Reg0C).group(1)
            reg0D = re.match(DDS_REGEX, self.Reg0D).group(1)

        # Assign upper and lower frequency limits
        if len(reg0B) == 16:
            f_upper = np.round(int(reg0B[0:8], 16) / (2**32) * self.DEFAULTS["dds_sys_clock_frequency"])
            f_lower = np.round(int(reg0B[8:16], 16) / (2**32) * self.DEFAULTS["dds_sys_clock_frequency"])
        else:
            raise ValueError("Invalid DDS Reg0B length ({:d}{:s}).".format(len(reg0B), reg0B))

        # Assign frequency steps
        if len(reg0C) == 16:
            df_negative = int(reg0C[0:8], 16) / (2**32) * self.DEFAULTS["dds_sys_clock_frequency"]
            df_positive = int(reg0C[8:16], 16) / (2**32) * self.DEFAULTS["dds_sys_clock_frequency"]
        else:
            raise ValueError("Invalid DDS Reg0C length ({:d}:{:s}).".format(len(reg0C), reg0C))
        
        # Assign time steps
        if len(reg0D) == 8:
            dt_negative = int(reg0D[0:4], 16) / self.DEFAULTS["dds_sys_clock_frequency"] * 4
            dt_positive = int(reg0D[4:8], 16) / self.DEFAULTS["dds_sys_clock_frequency"] * 4
        else:
            raise ValueError("Invalid DDS Reg0D length ({:d}:{:s}).".format(len(reg0D), reg0D))

        # Calculate center frequency
        fc = np.round((f_upper + f_lower) / 2)
    
        # Calculate bandwidth
        B = np.round(f_upper - f_lower)

        # Calculate period
        T = B / df_positive * dt_positive

        try:
            if self.SamplingFreqMode == 1:
                fs = 80e3 
            else:
                fs = 40e3
        except AttributeError:
            # Default to 40 kHz
            fs = 40e3

        self._add_immutable_attribute('T', T)
        self._add_immutable_attribute('B', B)
        self._add_immutable_attribute('fc', fc)
        self._add_immutable_attribute('fs', fs)

    def determine_file_format_version(self):
        """
        Determine the file format version from the read header lines

        The header lines are inspected to determine the file format version,
        and the DEFAULTS parsing tokens are setup accordingly
        """

        # Create a list of the first data dimension key from the alternative
        # file formats
        dim_keys = [item['data_dim_keys'][0] for item in self.ALTERNATIVES]

        # Search the header lines for one of the keys, to determine version
        for line in self.header_lines:
            for i, key in enumerate(dim_keys):
                if re.match(key, line):
                    for key in self.ALTERNATIVES[i]:
                        self.DEFAULTS[key] = self.ALTERNATIVES[i][key]
                    return

    def store_header(self):
        """
        Store the read header lines as parsed key/value pairs

        :returns: The parsed header
        :rtype: dict
        """

        self.header = {}

        for line in self.header_lines:
            item = line.split(self.DEFAULTS['header_line_delim'], maxsplit=1)

            if len(item) > 1 and item[0]:
                self.header[item[0].strip()] = item[1].strip()

        return self.header

    def define_data_shape(self):
        """
        Parse the data dimensions from the header to define the data shape

        If the Average header item is non-zero, then that indicates that the
        subbursts have been aggregated to a single value, so we rewrite that
        dimension as 1

        :returns: The data shape
        :rtype: tuple
        """

        self.data_shape = ()
        data_shape = []

        for key in self.DEFAULTS['data_dim_keys']:
            data_shape.append(int(self.header[key]))

        if int(self.header[self.DEFAULTS['data_type_key']]) > 0:
            data_shape[0] = 1

        self.data_shape = tuple(data_shape)

        return self.data_shape

    def define_data_type(self):
        """
        The Average item from the header is used to define the data type

        Average = 0: Record contains all subbursts as 16-bit ints
        Average = 1: Record contains the averaged subbursts as a 16-bit int
        Average = 2: Record contains the stacked subbursts as a 32-bit int

        :returns: The data type
        :rtype: str
        """

        try:
            self.data_type = self.DEFAULTS['data_types'][int(self.header[self.DEFAULTS['data_type_key']])]
        except IndexError as e:
            raise IndexError('Unsupported Averaging/Stacking configuration option found in header: {}'.format(int(self.header[self.DEFAULTS['data_type_key']])))

        return self.data_type

    def configure_from_header(self):
        """
        Configure the object from the raw header lines

        :returns: The parsed header
        :rtype: dict
        """

        if self.DEFAULTS['autodetect_file_format_version']:
            self.determine_file_format_version()

        self.store_header()
        self.define_data_shape()
        self.define_data_type()

        return self.header

    def read_header(self):
        """
        Read the header from the file

        * The raw header lines are available in the header_lines list
        * The parsed header is available in the header dict
        * The file offset to the start of the data is available in data_start
        * The data shape is available in the data_shape tuple
        * The data type is available in data_type

        :returns: The parsed header
        :rtype: dict
        """

        # The header lines are used to configure this object
        self.read_header_lines()
        self.configure_from_header()
        
        self.parse_header_lines()
        # Parse DDS registers from header values
        self.parse_dds_registers()

        return self.header

    def reshape_data(self):
        """
        Reshape the data according to the data_shape tuple

        If the data read from the file don't conform to the expected
        data_shape as determined from the header, then when reshaping the data,
        an exception will occur:

        * If the reshaping fails because the data array is shorter than
        expected, then we will reraise the exception.
        * If forgive mode is True, then we will emit a warning and then
        truncate the data to conform to the data_shape, otherwise we will
        reraise the exception.

        :returns: The data
        :rtype: numpy.array
        """

        try:
            self.data = np.reshape(self.data, self.data_shape, order=self.DEFAULTS['data_dim_order'])
        except ValueError as e:
            expected_len = self.data_shape[0] * self.data_shape[1]

            if self.data.size < expected_len:
                warnings.warn("Data array read from file doesn't match data_shape as read from the file header: {}. It is shorter than expected. Cannot continue.")
                raise
            elif self.data.size > expected_len:
                warnings.warn("Data array read from file doesn't match data_shape as read from the file header. It is longer than expected.")

                if self.DEFAULTS['forgive']:
                    warnings.warn("{}. Will truncate data to fit to data_shape.".format(e))
                    self.data = np.reshape(self.data[:expected_len], self.data_shape, order=self.DEFAULTS['data_dim_order'])
                else:
                    raise

        return self.data

    def read_data(self):
        """
        Read the data from the file

        The data are available in the data array, shaped according to the
        data_shape tuple

        :returns: The data
        :rtype: numpy.array
        """

        if self.data_start == -1:
            self.read_header()

        count = self.data_shape[0] * self.data_shape[1]
        self.fp.seek(self.data_start, 0)
        self.data = np.fromfile(self.fp, dtype=np.dtype(self.data_type), count=count)
        self.reshape_data()

        return self.data

    def format_header_line(self, key, value):
        """
        Format a raw header line from the given key and value

        :param key: The header item key
        :type key: str
        :param value: The header item value
        :type value: str
        :returns: The formatted raw header line
        :rtype: str
        """

        return '{}{}{}'.format(key, self.DEFAULTS['header_line_delim'], value)

    def reconstruct_header_lines(self):
        """
        Reconstruct the raw header lines from the parsed header

        :returns: The reconstructed raw header lines
        :rtype: list
        """

        self.header_lines = [self.format_header_line(k,v) for k,v in self.header.items()]
        self.header_lines.insert(0, self.DEFAULTS['header_markers']['start'])
        self.header_lines.append(self.DEFAULTS['header_markers']['end'])

        return self.header_lines

    def write_header(self, fp, subbursts=None, samples=None):
        """
        Write the header to the given file object

        :param fp: The open file object of the output file
        :type fp: file object
        :param subbursts: A range specifying the subbursts to be written
        :type subbursts: range object
        :param samples: A range specifying the samples to be written
        :type samples: range object
        """

        if self.data_start == -1:
            self.read_header()

        eol = self.DEFAULTS['header_line_eol']

        for line in self.header_lines:
            # Ensure requested subbursts & samples are reflected in the header
            if subbursts and re.match(self.DEFAULTS['data_dim_keys'][0], line):
                line = self.format_header_line(self.DEFAULTS['data_dim_keys'][0], len(subbursts))
            if samples and re.match(self.DEFAULTS['data_dim_keys'][1], line):
                line = self.format_header_line(self.DEFAULTS['data_dim_keys'][1], len(samples))

            fp.write(line + eol)

    def write_data(self, fp, subbursts=None, samples=None):
        """
        Write the data to the given file object

        :param fp: The open file object of the output file
        :type fp: file object
        :param subbursts: A range specifying the subbursts to be written
        :type subbursts: range object
        :param samples: A range specifying the samples to be written
        :type samples: range object
        """

        if self.data_start == -1:
            self.read_data()

        if not subbursts:
            subbursts = range(self.data_shape[0])
        if not samples:
            samples = range(self.data_shape[1])

        fp.write(np.asarray(self.data[subbursts.start:subbursts.stop:subbursts.step, samples.start:samples.stop:samples.step], order=self.DEFAULTS['data_dim_order']))

class ApRESFile(object):
    """
    Context manager for reading and writing ApRES files
    """

    DEFAULTS = {
        'file_encoding': 'latin-1',
        'apres_suffix': '.dat',
        'netcdf_suffix': '.nc',
        'netcdf_add_history': True,
        'netcdf_group_name': 'burst{:d}',
        'netcdf_var_name': 'data',
        'netcdf_attrs': {
            'units': '1',
            'long_name': 'Sub Burst ADC Samples'
        }
    }

    def __init__(self, path=None):
        """
        Constructor

        :param path: Path to the file
        :type path: str
        """

        self.path = path
        self.fp = None
        self.file_size = -1
        self.bursts = []

    def __enter__(self):
        """
        Enter the runtime context for this object

        The file is opened

        :returns: This object
        :rtype: ApRESFile
        """

        return self.open(self.path)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Exit the runtime context for this object

        The file is closed

        :returns: False
        :rtype: bool
        """

        self.close()

        return False         # This ensures any exception is re-raised

    def open(self, path=None, mode='r'):
        """
        Open the given file

        :param path: Path to the file
        :type path: str
        :param mode: Mode in which to open the file
        :type mode: str
        :returns: This object
        :rtype: ApRESFile
        """

        if path:
            self.path = path

        self.fp = open(self.path, mode, encoding=self.DEFAULTS['file_encoding'])
        self.file_size = os.fstat(self.fp.fileno()).st_size

        return self

    def close(self):
        """
        Close the file

        :returns: This object
        :rtype: ApRESFile
        """

        self.fp.close()
        self.fp = None

        return self

    def eof(self):
        """
        Report whether the end-of-file has been reached

        :returns: True if EOF has been reached, False otherwise
        :rtype: bool
        """

        return self.fp.tell() >= self.file_size

    def read_burst(self):
        """
        Read a burst block from the file

        :returns: The burst
        :rtype: ApRESBurst
        """

        burst = ApRESBurst(fp=self.fp)
        burst.read_data()

        return burst

    def read(self):
        """
        Reads all burst blocks from the file

        The bursts are available in the bursts list

        :returns: The bursts
        :rtype: list
        """

        while not self.eof():
            burst = self.read_burst()
            self.bursts.append(burst)

        return self.bursts

    def to_apres_dat(self, path, bursts=None, subbursts=None, samples=None):
        """
        Write the bursts to the given file path as an ApRES .dat file

        When rewriting the ApRES data to a new file, the bursts, subbursts,
        and the ADC samples for those selected subbursts, can be subsetted.
        The bursts, subbursts, and samples keyword arguments must specify a
        range object

        :param path: The path of the output file
        :type path: str
        :param bursts: A range specifying the bursts to be written
        :type bursts: range object
        :param subbursts: A range specifying the subbursts to be written
        :type subbursts: range object
        :param samples: A range specifying the samples to be written
        :type samples: range object
        """

        if not self.bursts:
            self.read()

        if not bursts:
            bursts = range(len(self.bursts))

        # We append each burst, so ensure file is empty if it already exists
        with open(path, 'w') as fout:
            pass

        # The ApRES .dat file format is a mixed mode file.  The header is
        # text, and the data section is binary
        for burst in self.bursts[bursts.start:bursts.stop:bursts.step]:
            with open(path, 'a') as fout:
                burst.write_header(fout, subbursts=subbursts, samples=samples)

            with open(path, 'ab') as fout:
                burst.write_data(fout, subbursts=subbursts, samples=samples)

    def burst_to_nc_object(self, burst, nco):
        """
        Map the given burst to the given netCDF object

        The netCDF object can either be a Dataset or a Group

        :param burst: The burst
        :type burst: ApRESBurst
        :param nco: A netCDF object (one of Dataset or Group)
        :type nco: netCDF4._netCDF4.Dataset or netCDF4._netCDF4.Group
        :returns: The netCDF object
        :rtype: netCDF4._netCDF4.Dataset or netCDF4._netCDF4.Group
        """

        # Write the burst header items as netCDF object attributes
        for key in burst.header:
            nco.setncattr(key, burst.header[key])

        for j, key in enumerate(burst.DEFAULTS['data_dim_keys']):
            nco.createDimension(key, burst.data_shape[j])

        data = nco.createVariable(self.DEFAULTS['netcdf_var_name'], burst.data_type, tuple(burst.DEFAULTS['data_dim_keys']))
        data.setncatts(self.DEFAULTS['netcdf_attrs'])
        data[:] = burst.data

        return nco

    def nc_object_to_burst(self, nco):
        """
        Map the given netCDF object to an ApRESBurst object

        The netCDF object can either be a Dataset or a Group

        :param nco: A netCDF object (one of Dataset or Group)
        :type nco: netCDF4._netCDF4.Dataset or netCDF4._netCDF4.Group
        :returns: The burst object
        :rtype: ApRESBurst
        """

        # We make a copy, otherwise data is invalid after file is closed
        data = nco.variables[self.DEFAULTS['netcdf_var_name']][:]
        attrs = vars(nco).copy()

        burst = ApRESBurst()
        burst.data_start = 0              # Stop data being read from file

        # Remove any attributes that weren't part of the original burst's header
        try:
            del attrs['history']
        except KeyError:
            pass

        # Reconstruct the original burst from the parsed header and data.  We
        # initially set the header_lines to be the parsed header which allows us
        # to determine the burst format version, and thus setup the DEFAULTS
        # parsing tokens accordingly.  This then allows us to correctly
        # reconstruct the raw header lines with the appropriate header line
        # delimiter for the burst format version
        burst.data = data
        burst.header = attrs
        burst.header_lines = burst.header
        burst.determine_file_format_version()
        burst.reconstruct_header_lines()
        burst.configure_from_header()

        return burst

    def to_netcdf(self, path=None, mode='w'):
        """
        Write the bursts to the given file path as a netCDF4 file

        The default netCDF file path is the same as the input file, but with
        a .nc suffix

        :param path: The path of the output file
        :type path: str
        :param mode: Mode in which to open the file
        :type mode: str
        """

        if not self.bursts:
            self.read()

        path = path or os.path.splitext(self.path)[0] + self.DEFAULTS['netcdf_suffix']

        ncfile = Dataset(path, mode)

        # Add the command line invocation to global history attribute
        if self.DEFAULTS['netcdf_add_history']:
            ncfile.history = ' '.join(sys.argv)

        # Write each burst as a netCDF4/HDF5 group
        for i, burst in enumerate(self.bursts):
            group = ncfile.createGroup(self.DEFAULTS['netcdf_group_name'].format(i))
            self.burst_to_nc_object(burst, group)

        ncfile.close()

