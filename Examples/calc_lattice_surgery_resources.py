"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2023 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""
#!/usr/bin/env python3
from __future__ import print_function

import os
import sys
import json

def extract_meas(meas):
    nq = len(meas)-1
    angle = meas['pi*']

    pauliS = ''
    for qi in range(nq):
        keyQ = 'q{}'.format(qi)
        pauliS += meas[keyQ]

    pWgt = sum(1 for c in pauliS if not c=='I')
    
    return angle, pauliS, pWgt

fname = sys.argv[1]
f = open(fname)

data = json.load(f)
ls_trans_data = data['3. Circuit after T depth reduction']
t_layers = ls_trans_data['T layers']

pWgtDict = dict()
for li, layer in enumerate(t_layers):
    print("expanding layer {}".format(li))
    for meas in layer:
        angle, pauliS, pWgt = extract_meas(meas)
        print("rot({}, {}):\t Pauli = {}".format(angle, pWgt, pauliS))
        if not pWgt in pWgtDict.keys():
            pWgtDict[pWgt] = 0
        pWgtDict[pWgt] += 1
maxWgt = max(pWgtDict.keys())

print("T count = {}, T depth = {}, max Pauli Wgt = {}".format(ls_trans_data['T count'], ls_trans_data['T depth'], maxWgt))
print("Pauli Weight distribution = {}".format(pWgtDict))
        
        


