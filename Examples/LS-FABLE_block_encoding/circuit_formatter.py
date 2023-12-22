#
# Copyright(c) Daniel Knuettel
#

#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#    Dieses Programm ist Freie Software: Sie können es unter den Bedingungen
#    der GNU General Public License, wie von der Free Software Foundation,
#    Version 3 der Lizenz oder (nach Ihrer Wahl) jeder neueren
#    veröffentlichten Version, weiterverbreiten und/oder modifizieren.
#
#    Dieses Programm wird in der Hoffnung, dass es nützlich sein wird, aber
#    OHNE JEDE GEWÄHRLEISTUNG, bereitgestellt; sogar ohne die implizite
#    Gewährleistung der MARKTFÄHIGKEIT oder EIGNUNG FÜR EINEN BESTIMMTEN ZWECK.
#    Siehe die GNU General Public License für weitere Details.
#
#    Sie sollten eine Kopie der GNU General Public License zusammen mit diesem
#    Programm erhalten haben. Wenn nicht, siehe <http://www.gnu.org/licenses/>.

"""
This module provides a class that can be used to embed
quantum circuits in Jupyter Notebooks.

The module requires that a pdflatex (preferably xelatex)
distribution and imagemagick is installed on the server.


Use it as::

    from circuit_formatter import CircuitPNGFormatter

    circuit = r'''
    \Qcircuit @C=1em @R=.7em {
    & \qw        & \ctrl{1} & \gate{c} & \ctrl{1} & \gate{R_X} & \ctrl{1} & \gate{a} & \qw \\
    & \gate{R_X} & \gate{X} & \gate{d} & \gate{X} & \gate{R_Z} & \gate{X} & \gate{b}& \qw \\
    }
    '''

    CircuitPNGFormatter(circuit)


"""

import os
import subprocess
import shutil
from collections import deque
from tempfile import TemporaryDirectory

class CircuitPNGFormatter(object):
    def __init__(self, circuit
                    , pdflatex="xelatex"
                    , convert="convert"
                    , pdflatex_args=["main.tex"]
                    , convert_args=["-profile", "\"icc\""
                                , "-density", "300"
                                , "main.pdf"
                                , "-quality", "90"
                                , "main.png"]
                    ):
        self.tex = circuit
        self.pdflatex = pdflatex
        self.convert = convert
        self.convert_args = convert_args
        self.pdflatex_args = pdflatex_args

        if(shutil.which(self.pdflatex) is None):
            raise OSError(f"pdflatex ({self.pdflatex}) not found, set it using ``pdflatex=<program>``")
        if(shutil.which(self.convert) is None):
            raise OSError(f"imagemagick ({self.convert}) not found, set it using ``convert=<program>``")

    def _repr_png_(self):
        with TemporaryDirectory() as tmpdirname:
            with open(tmpdirname + "/main.tex", "w") as fout:
                fout.write(self.get_tex_file_content())
            subprocess.run([self.pdflatex] + self.pdflatex_args, cwd=tmpdirname)
            if(not os.path.isfile(tmpdirname + "/main.pdf")):
                raise OSError(f"pdflatex ({self.pdflatex}) did not produce a pdf file")
            subprocess.run([self.convert] + self.convert_args
                            , cwd=tmpdirname)
            if(not os.path.isfile(tmpdirname + "/main.png")):
                raise OSError(f"imagemagick ({self.convert}) did not produce a png file")
            with open(tmpdirname + "/main.png", "rb") as fin:
                return fin.read()

    def get_tex_file_content(self):
        header = r'''
        \documentclass[preview]{standalone}
        \usepackage[utf8]{inputenc}
        \usepackage[T1]{fontenc}

        \usepackage{qcircuit}

        \title{Drawing Circuits with qcircuit}

        \begin{document}
        '''
        bottom = r'''

        \end{document}

        '''

        return header + self.tex + bottom



