import argparse

from apres import ApRESFile
import apres.__main__ as m

PROGNAME = 'read_apres'

def main():
    parser = argparse.ArgumentParser(description='print the header, a sample of the data, and diagnostics, from the given raw ApRES file', prog=PROGNAME)
    parser.add_argument('infile', help='ApRES raw file')
    m._add_fsspec_opts(parser)
    m._add_common_opts(parser)
    args = parser.parse_args()

    with ApRESFile(args.infile, fs_opts=args.fs_opts) as f:
        for burst in f.read():
            header = burst.header
            print(f'header: header_start = {burst.header_start}')
            print(f'header: data_start = {burst.data_start}')
            print(f'header: data_dim_keys = {burst.data_dim_keys}')
            print(f'header: data_shape = {burst.data_shape}')
            print(f'header: dict = ')
            print(header)

            data = burst.data
            ndims = len(data.shape)
            print(f'data: len(data) = {len(data)}')
            print(f'data: data.size = {data.size}')
            print(f'data: data.shape = {data.shape}')

            # Show a small selection of the first subbursts and samples.
            # If this is a multi-attenuator dataset, show data for each one
            if len(data) > 0:
                nsubbursts = 10 if data.shape[0] > 10 else data.shape[0]

                if ndims > 2:
                    nsamples = 10 if data.shape[2] > 10 else data.shape[2]
                    data_slice = data[0:nsubbursts, :, 0:nsamples]
                else:
                    nsamples = 10 if data.shape[1] > 10 else data.shape[1]
                    data_slice = data[0:nsubbursts, 0:nsamples]

                print('data: head = ')
                print(data_slice)

        print(f'number of bursts = {len(f.bursts)}')

if __name__ == '__main__':
    main()

