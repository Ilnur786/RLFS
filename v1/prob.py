import os

os.environ['DB'] = './data/db.shelve'
os.environ['DATA'] = './data/rules'

from RNNSynthesis import SimpleSynthesis
from CGRtools.files import SDFRead
from RNNSynthesis.helper import get_feature, get_feature_bits
from MorganFingerprint import MorganFingerprint

with SDFRead('../data/tylenol.sdf', 'rb') as f:
    mc = next(f)


print(mc)
print(type(mc))
fp = get_feature(mc)
fp1 = get_feature_bits(mc)
print(len(fp))
print(fp1.size)
print(type(fp1))
print(fp1)
# C1(=C2C(C=C(C(=C2)NC(C=C)=O)OCCNC(CC)=O)=NC=N1)NC=3C(=CC(=CC=3)Br)F
# <class 'CGRtools.containers.molecule.MoleculeContainer'>
# 542
# 8192 state_size
