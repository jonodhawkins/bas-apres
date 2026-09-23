import argparse

from apres import ApRESFile
import apres.__main__ as m

PROGNAME = 'write_apres'

def main():
    parser = argparse.ArgumentParser(description='read a raw ApRES file and write to another raw ApRES file, optionally subsetting the data', prog=PROGNAME)
    parser.add_argument('infile', help='ApRES raw file')
    parser.add_argument('outfile', help='ApRES raw file')
    parser.add_argument('-b', '--bursts', help='only write out the first BURSTS bursts', dest='bursts', default=None, type=int)
    parser.add_argument('-u', '--subbursts', help='only write out the first SUBBURSTS subbursts', dest='subbursts', default=None, type=int)
    parser.add_argument('-s', '--samples', help='only write out the first SAMPLES ADC samples', dest='samples', default=None, type=int)
    m._add_fsspec_opts(parser)
    m._add_common_opts(parser)
    args = parser.parse_args()

    # We can optionally select a subset of bursts, subbursts and/or samples
    # for output.  These have to be range objects
    if args.bursts:
        args.bursts = range(args.bursts)
    if args.subbursts:
        args.subbursts = range(args.subbursts)
    if args.samples:
        args.samples = range(args.samples)
 
    with ApRESFile(args.infile, fs_opts=args.fs_opts) as f:
        f.to_apres_dat(args.outfile, bursts=args.bursts, subbursts=args.subbursts, samples=args.samples)

if __name__ == '__main__':
    main()

