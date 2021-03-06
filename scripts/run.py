#!/usr/bin/env python

import logging
import sys
import os
import inspect
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
cmd_folder = os.path.realpath(os.path.join(cmd_folder, ".."))
if cmd_folder not in sys.path:
    sys.path.insert(0,cmd_folder)
    
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    
from dlbbo.dlbbo import DLBBO
    
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-s", "--scenario", help="path to scenario infos")
    parser.add_argument("-m", "--model_type", choices=["VGG-like","MobileNet", "VGG16"], help="path to scenario infos")
    parser.add_argument("-v", "--verbose", type=int, help="verbosity of keras.fit (0, 1 or 2)")

    args_ = parser.parse_args()

    dlbbo = DLBBO(scenario_dn=args_.scenario,
                  model_type=args_.model_type,
                  verbose=args_.verbose)
    dlbbo.main()
