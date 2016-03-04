#!/usr/bin/env python
from os import environ, path

from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *

MODELDIR = "/tmp2/Data/asr_model"
DATADIR = "/tmp2/Data"


# Create a decoder with certain model
config = Decoder.default_config()
config.set_string('-hmm', path.join(MODELDIR, 'en-us'))
config.set_string('-lm', path.join(MODELDIR, 'en-us.lm'))
config.set_string('-dict', path.join(MODELDIR, 'en-us.dict'))
decoder = Decoder(config)






