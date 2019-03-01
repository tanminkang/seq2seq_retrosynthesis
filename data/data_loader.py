#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:00:29 2019

@author: tanminkang
"""
import numpy as np
import pandas as pd

file_path = '../../raw/'
file_names = ['1976_Sep2016_USPTOgrants_smiles.rsmi','2001_Sep2016_USPTOapplications_smiles.rsmi']

df_1976 = pd.read_csv(file_path+file_names[0], sep='\t')
df_2001 = pd.read_csv(file_path+file_names[1], sep='\t')
df = df_1976.append(df_2001, ignore_index=True)

train = df_1976[""]
test = df_2001[""]




