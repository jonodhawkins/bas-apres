import unittest
from unittest.mock import Mock, patch
import datetime
import os
import warnings
import tempfile
import hashlib
 
import numpy as np
from netCDF4 import Dataset

from apres import __version__, ApRESBurst, ApRESFile

def test_version():
    assert __version__ == '0.1.1'

class TestApRESBurst(unittest.TestCase):

    base = os.path.dirname(__file__)

    def test_optional_fp_in_constructor(self):
        f = ApRESBurst()

    def test_store_header_ok(self):
        f = ApRESBurst()
        f.header_lines = ['Temp1=10.0469','Temp2=10.1094','BatteryVoltage=12.2058']
        f.store_header()
        self.assertEqual('10.0469', f.header['Temp1'])
        self.assertEqual('10.1094', f.header['Temp2'])
        self.assertEqual('12.2058', f.header['BatteryVoltage'])

    def test_store_header_value_has_delimiter(self):
        f = ApRESBurst()
        f.header_lines = ['Dummy=value=10']
        f.store_header()
        self.assertEqual('value=10', f.header['Dummy'])

    def test_store_header_no_value(self):
        f = ApRESBurst()
        f.header_lines = ['Dummy=']
        f.store_header()
        self.assertEqual('', f.header['Dummy'])

    def test_store_header_no_key(self):
        f = ApRESBurst()
        f.header_lines = ['=12']
        f.store_header()
        self.assertNotIn('', f.header)

    def test_store_header_false_like_key(self):
        f = ApRESBurst()
        f.header_lines = ['0=12','False=0']
        f.store_header()
        self.assertEqual('12', f.header['0'])
        self.assertEqual('0', f.header['False'])

    def test_store_header_invalid_delimiter(self):
        f = ApRESBurst()
        f.header_lines = ['Dummy;10']
        f.store_header()
        self.assertNotIn('Dummy', f.header)

    def test_store_header_strip_whitespace(self):
        f = ApRESBurst()
        f.header_lines = ['  Temp1  =  10.0469  ','  Temp2  =  10.1094  ']
        f.store_header()
        self.assertEqual('10.0469', f.header['Temp1'])
        self.assertEqual('10.1094', f.header['Temp2'])

    def test_determine_file_format_version_ok(self):
        f = ApRESBurst()
        f.header_lines = ['NSubBursts=100','N_ADC_SAMPLES=40001']
        f.determine_file_format_version()
        self.assertEqual('=', f.DEFAULTS['header_line_delim'])
        self.assertEqual(['NSubBursts', 'N_ADC_SAMPLES'], f.DEFAULTS['data_dim_keys'])

        f.header_lines = ['SubBursts in burst: 100', 'Samples: 40001']
        f.determine_file_format_version()
        self.assertEqual(':', f.DEFAULTS['header_line_delim'])
        self.assertEqual(['SubBursts in burst', 'Samples'], f.DEFAULTS['data_dim_keys'])

    def test_determine_file_format_version_turned_off(self):
        f = ApRESBurst()
        f.reset_init_defaults()

        # f.DEFAULTS['autodetect_file_format_version'] = False
        f.header_lines = ['SubBursts in burst: 100', 'Samples: 40001']
        # f.determine_file_format_version()
        #self.assertEqual(':', f.DEFAULTS['header_line_delim'])
        #self.assertEqual(['SubBursts in burst', 'Samples'], f.DEFAULTS['data_dim_keys'])

    def test_define_data_shape_ok(self):
        f = ApRESBurst()
        f.header_lines = ['NSubBursts=100','N_ADC_SAMPLES=40001','Average=0']
        f.store_header()
        f.define_data_shape()
        self.assertEqual((100,40001), f.data_shape)

    def test_define_data_shape_non_integer_value(self):
        f = ApRESBurst()
        f.header_lines = ['NSubBursts=10.37','N_ADC_SAMPLES=40001','Average=0']
        f.store_header()

        with self.assertRaises(ValueError):
            f.define_data_shape()

    def test_define_data_shape_no_value(self):
        f = ApRESBurst()
        f.header_lines = ['NSubBursts=','N_ADC_SAMPLES=40001','Average=0']
        f.store_header()

        with self.assertRaises(ValueError):
            f.define_data_shape()

    def test_define_data_shape_invalid_delimiter(self):
        f = ApRESBurst()
        f.header_lines = ['NSubBursts;100','N_ADC_SAMPLES=40001','Average=0']
        f.store_header()

        with self.assertRaises(KeyError):
            f.define_data_shape()

    def test_define_data_shape_required_key_missing(self):
        f = ApRESBurst()
        f.header_lines = ['NSubBursts_is_missing=100','N_ADC_SAMPLES=40001','Average=0']
        f.store_header()

        with self.assertRaises(KeyError):
            f.define_data_shape()

    def test_define_data_type_ok(self):
        f = ApRESBurst()
        f.header_lines = ['Average=0']
        f.store_header()
        f.define_data_type()
        self.assertEqual(f.DEFAULTS['data_types'][0], f.data_type)

    def test_define_data_type_non_integer_value(self):
        f = ApRESBurst()
        f.header_lines = ['Average=0.5']
        f.store_header()

        with self.assertRaises(ValueError):
            f.define_data_type()

    def test_define_data_type_unsupported_value(self):
        f = ApRESBurst()
        f.header_lines = ['Average=3']
        f.store_header()

        with self.assertRaises(IndexError):
            f.define_data_type()

    def test_define_data_type_required_key_missing(self):
        f = ApRESBurst()
        f.header_lines = ['Average_is_missing=0']
        f.store_header()

        with self.assertRaises(KeyError):
            f.define_data_type()

    def test_read_header_ok(self):
        in_file = self.base + '/short-test-data.dat'

        with open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding']) as fp:
            f = ApRESBurst(fp)
            f.read_header()

    def test_read_header_invalid_file(self):
        in_file = self.base + '/non-existent-file'

        with self.assertRaises(FileNotFoundError):
            with open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding']) as fp:
                f = ApRESBurst(fp)
                f.read_header()

    def test_read_header_configures_object(self):
        in_file = self.base + '/short-test-data.dat'

        fp = open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding'])
        f = ApRESBurst(fp)
        f.read_header()
        self.assertEqual(804, f.data_start)
        self.assertEqual((1, 500), f.data_shape)
        self.assertEqual('10.0469', f.header['Temp1'])
        self.assertEqual('10.1094', f.header['Temp2'])
        self.assertEqual('12.2058', f.header['BatteryVoltage'])
        fp.close()

    def test_read_data_ok(self):
        in_file = self.base + '/short-test-data.dat'

        with open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding']) as fp:
            f = ApRESBurst(fp)
            f.read_data()

    def test_read_data_invalid_file(self):
        in_file = self.base + '/non-existent-file'

        with self.assertRaises(FileNotFoundError):
            with open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding']) as fp:
                f = ApRESBurst(fp)
                f.read_data()

    def test_read_data_reads_header_first_if_not_already_done(self):
        in_file = self.base + '/short-test-data.dat'

        fp = open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding'])
        f = ApRESBurst(fp)
        self.assertEqual(-1, f.data_start)
        self.assertEqual(0, len(f.header))
        f.read_data()
        self.assertNotEqual(-1, f.data_start)
        self.assertNotEqual(0, len(f.header))
        fp.close()

    def test_read_data_does_not_read_header_if_already_done(self):
        in_file = self.base + '/short-test-data.dat'

        fp = open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding'])
        f = ApRESBurst(fp)

        # Setting an arbitrary file offset position for the start of the data
        # section, will mess up reading the data.  The data cannot be reshaped,
        # because data_shape is an empty tuple.  We catch the ValueError thrown
        # when reshaping, because we check to see if the data are shorter or
        # longer than expected.  As the data_shape tuple is empty though, we
        # will cause an IndexError
        f.data_start = 7

        with self.assertRaises(IndexError):
            f.read_data()

        self.assertEqual(0, len(f.data_shape))
        self.assertEqual(0, len(f.header))
        fp.close()

    def test_read_data_shapes_the_data_according_to_header(self):
        in_file = self.base + '/short-test-data.dat'

        fp = open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding'])
        f = ApRESBurst(fp)
        f.read_data()
        self.assertEqual(f.data_shape, f.data.shape)
        self.assertEqual(f.data_shape[0] * f.data_shape[1], f.data.size)
        fp.close()

    def test_read_data_fails_to_shape_shorter_data(self):
        in_file = self.base + '/short-test-data.dat'

        fp = open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding'])
        f = ApRESBurst(fp)
        f.read_data()

        # Change the header so we're expecting twice as much data as we
        # actually read.  We catch the initial ValueError, so that we can
        # provide a meaningful warning, then reraise the error
        f.header['NSubBursts'] = 2
        f.define_data_shape()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with self.assertRaises(ValueError):
                f.reshape_data()

        fp.close()

    def test_read_data_forgives_to_shape_longer_data(self):
        in_file = self.base + '/short-test-data.dat'

        fp = open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding'])
        f = ApRESBurst(fp)
        f.read_data()

        # When calling read_data(), the data are shaped according to the
        # header.  To redefine the shape and truncate, we need to effectively
        # read the data as a 1D linear array, as they are when first read
        f.data_shape = (f.data.size)
        f.reshape_data()

        # Change the header so we're expecting half as much data as we
        # actually read.  We catch the initial ValueError, so that we can
        # provide a meaningful warning.  If in forgive mode, we truncate the
        # data and continue without error
        f.header['N_ADC_SAMPLES'] = 250
        f.define_data_shape()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f.reshape_data()
            self.assertEqual(f.data_shape, f.data.shape)
            self.assertEqual(f.data_shape[0] * f.data_shape[1], f.data.size)

        fp.close()

    def test_read_data_fails_to_shape_longer_data(self):
        in_file = self.base + '/short-test-data.dat'

        fp = open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding'])
        f = ApRESBurst(fp)
        f.read_data()

        # Reset data shape as 1D
        f.data_shape = (f.data.size)
        f.reshape_data()

        # Change the header so we're expecting half as much data as we
        # actually read.  We catch the initial ValueError, so that we can
        # provide a meaningful warning.  If not in forgive mode, we reraise
        # the error
        f.header['N_ADC_SAMPLES'] = 250
        f.define_data_shape()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            with self.assertRaises(ValueError):
                with patch.dict(f.DEFAULTS, {'forgive': False}):
                    f.reshape_data()

        fp.close()

    def test_read_data_reads_expected_values(self):
        in_file = self.base + '/short-test-data.dat'

        fp = open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding'])
        f = ApRESBurst(fp)
        f.read_data()
        self.assertEqual(33774, f.data[0, 0])
        fp.close()

    def test_format_header_line_ok(self):
        f = ApRESBurst()

        # Ensure this is the original default, as it may have been changed by
        # previous tests
        f.DEFAULTS['header_line_delim'] = '='

        s = f.format_header_line('NSubBursts', '1')
        self.assertEqual('NSubBursts=1', s)

    def test_format_header_line_modified_delimiter(self):
        f = ApRESBurst()

        with patch.dict(f.DEFAULTS, {'header_line_delim': ':'}):
            s = f.format_header_line('NSubBursts', '1')
            self.assertEqual('NSubBursts:1', s)

    def test_write_header_ok(self):
        in_file = self.base + '/short-test-data.dat'
 
        fp = open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding'])
        fin = ApRESBurst(fp)

        # We mock the call to write() to avoid writing an output file
        fout = Mock(write=Mock())

        fin.write_header(fout)
        fout.write.assert_any_call('NSubBursts=1\r\n')
        fp.close()

    def test_write_header_reads_header_first_if_not_already_done(self):
        in_file = self.base + '/short-test-data.dat'
 
        fp = open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding'])
        fin = ApRESBurst(fp)

        fout = Mock(write=Mock())

        self.assertEqual(-1, fin.data_start)
        self.assertEqual(0, len(fin.header))
        fin.write_header(fout)
        self.assertNotEqual(-1, fin.data_start)
        self.assertNotEqual(0, len(fin.header))
        fp.close()

    def test_write_header_modified_eol(self):
        in_file = self.base + '/short-test-data.dat'

        fp = open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding'])
        fin = ApRESBurst(fp)

        fout = Mock(write=Mock())

        with patch.dict(fin.DEFAULTS, {'header_line_eol': '\n'}):
            fin.write_header(fout)
            fout.write.assert_any_call('NSubBursts=1\n')

        fp.close()

    def test_write_header_rewrite_dimensions(self):
        in_file = self.base + '/short-test-data.dat'

        fp = open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding'])
        fin = ApRESBurst(fp)
 
        fout = Mock(write=Mock())

        fin.write_header(fout, samples=range(10))
        fout.write.assert_any_call('N_ADC_SAMPLES=10\r\n')

        fp.close()

    def test_write_header_rewrite_dimensions_invalid_kwarg_type(self):
        in_file = self.base + '/short-test-data.dat'

        fp = open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding'])
        fin = ApRESBurst(fp)
 
        fout = Mock(write=Mock())

        with self.assertRaises(TypeError):
            # Keyword argument `samples` must be a range object
            fin.write_header(fout, samples=10)

        fp.close()

    def test_write_data_ok(self):
        in_file = self.base + '/short-test-data.dat'
 
        fp = open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding'])
        fin = ApRESBurst(fp)

        fout = Mock(write=Mock())

        fin.write_data(fout)
        self.assertEqual(fin.data.shape, fin.data_shape)
        fout.write.assert_called()
        fp.close()

    def test_write_data_reads_data_first_if_not_already_done(self):
        in_file = self.base + '/short-test-data.dat'
 
        fp = open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding'])
        fin = ApRESBurst(fp)

        fout = Mock(write=Mock())

        self.assertEqual(-1, fin.data_start)
        self.assertEqual(0, len(fin.header))
        fin.write_data(fout)
        self.assertNotEqual(-1, fin.data_start)
        self.assertNotEqual(0, len(fin.header))
        fp.close()

    def test_write_data_uses_default_dimensions(self):
        in_file = self.base + '/short-test-data.dat'

        fp = open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding'])
        fin = ApRESBurst(fp)
 
        fout = Mock(write=Mock())

        fin.write_data(fout)
        subbursts = range(fin.data_shape[0])
        samples = range(fin.data_shape[1])
        data_expected = np.asarray(fin.data[subbursts.start:subbursts.stop:subbursts.step, samples.start:samples.stop:samples.step], order=fin.DEFAULTS['data_dim_order'])
        data_actual = fout.write.call_args[0][0]
        self.assertTrue(np.array_equal(data_actual, data_expected))

        fp.close()

    def test_write_data_rewrite_dimensions(self):
        in_file = self.base + '/short-test-data.dat'
        samples = range(10)

        fp = open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding'])
        fin = ApRESBurst(fp)
 
        fout = Mock(write=Mock())

        fin.write_data(fout, samples=samples)
        subbursts = range(fin.data_shape[0])
        data_expected = np.asarray(fin.data[subbursts.start:subbursts.stop:subbursts.step, samples.start:samples.stop:samples.step], order=fin.DEFAULTS['data_dim_order'])
        data_actual = fout.write.call_args[0][0]
        self.assertTrue(np.array_equal(data_actual, data_expected))

        fp.close()

    def compare_reconstructed_header_lines(self, header, expected_header_lines):
        f = ApRESBurst()

        # We initially set the header_lines to be the parsed header which
        # allows us to determine the file format version, and thus setup the
        # DEFAULTS parsing tokens accordingly.  This then allows us to correctly
        # reconstruct the raw header lines with the appropriate header line
        # delimiter for the file format version
        f.header = header
        f.header_lines = f.header
        f.determine_file_format_version()
        f.reconstruct_header_lines()
        self.assertEqual(expected_header_lines, f.header_lines)

    def test_reconstruct_header_lines_v2(self):
        expected_header_lines = ['\r\n*** Burst Header ***', 'Time stamp=2019-12-25 03:26:37', 'NSubBursts=100', 'Average=0', 'N_ADC_SAMPLES=40001', '\r\n*** End Header ***']
        header = {'Time stamp': '2019-12-25 03:26:37', 'NSubBursts': '100', 'Average': '0', 'N_ADC_SAMPLES': '40001'}
        self.compare_reconstructed_header_lines(header, expected_header_lines)

    def test_reconstruct_header_lines_v1(self):
        expected_header_lines = ['\r\n*** Burst Header ***', 'Samples:60000', 'SubBursts in burst:100', 'Time stamp:2013-12-27 10:32:31', 'Average:2', '\r\n*** End Header ***']
        header = {'Samples': '60000', 'SubBursts in burst': '100', 'Time stamp': '2013-12-27 10:32:31', 'Average': '2'}
        self.compare_reconstructed_header_lines(header, expected_header_lines)

    def test_header_timestamp_conversion(self):
        in_file = self.base + '/short-test-data.dat'
        with open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding']) as fp:
            f = ApRESBurst(fp)
            f.read_header()
            self.assertTrue(isinstance(f.timestamp, datetime.datetime))
            self.assertEqual(f.timestamp, datetime.datetime(2014,12,12,19,42,6))

    def test_header_integer_conversion(self):
        in_file = self.base + '/short-test-data.dat'
        with open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding']) as fp:
            f = ApRESBurst(fp)
            f.read_header()
            self.assertTrue(isinstance(f.WATCHDOG_TASK_SECS, int))
            self.assertEqual(f.WATCHDOG_TASK_SECS, 3600)


    def test_immutable_parameter(self):        
        in_file = self.base + '/short-test-data.dat'
        with open(in_file, encoding=ApRESFile.DEFAULTS['file_encoding']) as fp:
            f = ApRESBurst(fp)
            f.read_header()
            print(f.nAttenuators)
            with self.assertRaises(AttributeError):
                f.nAttenuators = 10

class TestApRESFile(unittest.TestCase):

    base = os.path.dirname(__file__)

    def test_override_path_in_open(self):
        non_existent_file = self.base + '/non-existent-file'
        existent_file = self.base + '/short-test-data.dat'

        f = ApRESFile(non_existent_file)

        with self.assertRaises(FileNotFoundError):
            f.open()
            f.read()
            f.close()

        # Override the path given to the constructor with a valid path
        f.open(existent_file)
        f.read()
        f.close()

    def test_optional_path_in_open(self):
        in_file = self.base + '/short-test-data.dat'

        f = ApRESFile(in_file)
        f.open()
        f.read()
        f.close()

    def test_optional_path_in_constructor(self):
        in_file = self.base + '/short-test-data.dat'

        f = ApRESFile()
        f.open(in_file)
        f.read()
        f.close()

    def test_to_apres_dat_uses_default_dimensions(self):
        in_file = self.base + '/short-test-data-ts.dat'

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = tmp_dir + '/output.dat'

            with ApRESFile(in_file) as fin:
                fin.to_apres_dat(out_file)
                in_nbursts = len(fin.bursts)

            with ApRESFile(out_file) as fout:
                fout.read()
                out_nbursts = len(fout.bursts)

            assert in_nbursts == out_nbursts

    def test_to_apres_dat_rewrite_dimensions(self):
        in_file = self.base + '/short-test-data-ts.dat'
        bursts = range(3)

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = tmp_dir + '/output.dat'

            # Subset the input file as the first three bursts
            with ApRESFile(in_file) as fin:
                fin.to_apres_dat(out_file, bursts=bursts)
                in_nbursts = len(fin.bursts)

            with ApRESFile(out_file) as fout:
                fout.read()
                out_nbursts = len(fout.bursts)

            assert in_nbursts == 5
            assert out_nbursts == 3

class TestConversion(unittest.TestCase):

    base = os.path.dirname(__file__)

    def compare_original_with_recovered(self, in_file):
        with tempfile.TemporaryDirectory() as tmp_dir:
            nc_file = tmp_dir + '/converted.nc'
            recovered_file = tmp_dir + '/recovered.dat'

            # Convert to netCDF4
            with ApRESFile(in_file) as f:
                f.to_netcdf(nc_file)

            # Recover the ApRES file
            fout = ApRESFile()

            with Dataset(nc_file, 'r') as fin:
                for key in fin.groups:
                    burst = fout.nc_object_to_burst(fin.groups[key])
                    fout.bursts.append(burst)

            fout.to_apres_dat(recovered_file)

            # Compare the original and the recovered files
            with open(in_file, 'rb') as f:
                hash1 = hashlib.sha1(f.read()).hexdigest()

            with open(recovered_file, 'rb') as f:
                hash2 = hashlib.sha1(f.read()).hexdigest()

        return (hash1, hash2)

    def test_conversion_and_recovery_v2(self):
        in_file = self.base + '/short-test-data-v2.dat'
        hash1, hash2 = self.compare_original_with_recovered(in_file)
        assert hash1 == hash2

    def test_conversion_and_recovery_v1(self):
        # When converting v1 format files, the original and recovered are
        # identical except for trivial differences in whitespace in the header
        in_file = self.base + '/short-test-data-v1.dat'
        hash1, hash2 = self.compare_original_with_recovered(in_file)
        assert hash1 != hash2

