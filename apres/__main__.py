"""
Package for working with BAS ApRES files

This is the package main, however, there is no package main functionality.
See the individual CLI scripts (e.g. read_apres, plot_apres etc. instead.)
"""

import os
import json
import logging

from apres import __version__

LOGLEVEL = os.environ.get('LOGLEVEL', 'WARNING').upper()
logging.basicConfig(level=LOGLEVEL)

def _json_read(filename):
    with open(filename) as f:
        data = json.load(f)

    return data

def _add_fsspec_opts(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-o', '--fs-opts', help='fsspec filesystem options from a JSON string', default={}, type=json.loads, dest='fs_opts')
    group.add_argument('-O', '--fs-opts-file', help='fsspec filesystem options from a JSON file', default={}, type=_json_read, dest='fs_opts')

def _add_common_opts(parser):
    parser.add_argument('-V', '--version', action='version', version=f"%(prog)s {__version__}")

if __name__ == '__main__':
    print('There is no package main functionality.  See the individual CLI scripts (e.g. read_apres, plot_apres etc. instead.)')

