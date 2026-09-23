import argparse

from apres import ApRESFile
import apres.__main__ as m

PROGNAME = 'apres_to_nc'

def main():
    parser = argparse.ArgumentParser(description='convert a raw ApRES file to a netCDF4 file', prog=PROGNAME)
    parser.add_argument('infile', help='ApRES raw file')
    parser.add_argument('outfile', help='converted netCDF file', default=None, nargs='?')
    m._add_fsspec_opts(parser)
    m._add_common_opts(parser)
    args = parser.parse_args()
 
    with ApRESFile(args.infile, fs_opts=args.fs_opts) as f:
        f.to_netcdf(args.outfile)

if __name__ == '__main__':
    main()

