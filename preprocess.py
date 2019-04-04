#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 13:37:27 2019

@author: tanminkang
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import re

# Reactants and reagent separation(atom mappings)
def separate_reactant_reagent(smiles):
    """
    Reactants and reagent separation (atom mappings)
    :param smi:
    :return:
    """ 
    smiles = smiles.split(' |f')[0]# remove useless information
    reactant = smiles.split('>')[0]
    reagent = smiles.split('>')[1]
    product = smiles.split('>')[2]
    
    return reactant, reagent, product

def canonicalize(smiles): # canonicalize smiles by MolToSmiles function
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) if (smiles != '') else ''

def remove_atom_mapping(smi):
    """
    Atom-mapping removal and canonicalization
    :param smi:
    :return:
    """
    # canonicalization
    smi = re.sub(r'H[0-9]+|H|:[0-9]+', '', smi)
    myRe = re.compile(r"(\[)([A-Za-z]+)(\])")
    smi = myRe.sub(r'\2', smi)
    return smi

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def main():
    """
    """
    # Atom-mapping removal and canonicalization
    reactant, reagent, product = separate_reactant_reagent(smi)
    reactant = remove_atom_mapping(canonicalize(reactant))
    reagent = remove_atom_mapping(canonicalize(reagent))
    product = remove_atom_mapping(canonicalize(product))
    
    # Reactants and product tokenization
    Source = smi_tokenizer(reactant)+' > A_'+reagent# Reagent tokenization
    Target = smi_tokenizer(product)
