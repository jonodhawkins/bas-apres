import os
import argparse

from netCDF4 import Dataset

from apres import ApRESFile
import apres.__main__ as m

PROGNAME = 'nc_to_apres'

def main():
    parser = argparse.ArgumentParser(description='recover the raw ApRES file from a converted netCDF4 file', prog=PROGNAME)
    parser.add_argument('infile', help='converted netCDF file of raw ApRES data')
    parser.add_argument('outfile', help='ApRES raw file', default=None, nargs='?')
    m._add_common_opts(parser)
    args = parser.parse_args()

    # We can optionally be given an explicit output filename 
    if not args.outfile:
        args.outfile = os.path.splitext(args.infile)[0] + ApRESFile.DEFAULTS['apres_suffix']

    fout = ApRESFile()

    with Dataset(args.infile, 'r') as fin:
        # Convert each netCDF group to an ApRESBurst object
        for key in fin.groups:
            burst = fout.nc_object_to_burst(fin.groups[key])
            fout.bursts.append(burst)

    fout.to_apres_dat(args.outfile)

if __name__ == '__main__':
    main()

