#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:38:17 2019

@author: tanminkang
"""

from rdkit import Chem

smi = ''
random_equivalent_smiles = Chem.MolFromSmiles(Chem.MolToSmiles(smi, doRandom=True))