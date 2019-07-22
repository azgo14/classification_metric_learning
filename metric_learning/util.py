# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import torch


class SimpleLogger(object):
    def __init__(self, logfile, terminal):
        ZERO_BUFFER_SIZE = 0  # immediately flush logs

        self.log = open(logfile, 'a', ZERO_BUFFER_SIZE)
        self.terminal = terminal

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()