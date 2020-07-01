# -*- coding: utf-8 -*-
"""
Test data
"""

# Imports

import pandas as pd
from edbo.feature_utils import build_experiment_index

# Build data sets from indices

def aryl_amination(aryl_halide='ohe', additive='ohe', base='ohe', ligand='ohe', subset=1):
    """
    Load aryl amination data with different features.
    """
    
    # SMILES index
    
    index = pd.read_csv('data/aryl_amination/experiment_index.csv')
    
    # Choose supset:
    
    ar123 = ['FC(F)(F)c1ccc(Cl)cc1','FC(F)(F)c1ccc(Br)cc1','FC(F)(F)c1ccc(I)cc1']
    ar456 = ['COc1ccc(Cl)cc1','COc1ccc(Br)cc1','COc1ccc(I)cc1']
    ar789 = ['CCc1ccc(Cl)cc1','CCc1ccc(Br)cc1','CCc1ccc(I)cc1']
    ar101112 = ['Clc1ccccn1','Brc1ccccn1','Ic1ccccn1']
    ar131415 = ['Clc1cccnc1','Brc1cccnc1','Ic1cccnc1']
    
    def get_subset(ar):
        a = index[index['Aryl_halide_SMILES'] == ar[0]]
        b = index[index['Aryl_halide_SMILES'] == ar[1]]
        c = index[index['Aryl_halide_SMILES'] == ar[2]]
        return pd.concat([a,b,c])
        
    if subset == 1:
        index = get_subset(ar123)
    elif subset == 2:
        index = get_subset(ar456)
    elif subset == 3:
        index = get_subset(ar789)
    elif subset == 4:
        index = get_subset(ar101112)
    elif subset == 5:
        index = get_subset(ar131415)
    
    # Aryl halide features
    
    if aryl_halide == 'dft':
        aryl_features = pd.read_csv('data/aryl_amination/aryl_halide_dft.csv')
    elif aryl_halide == 'mordred':
        aryl_features = pd.read_csv('data/aryl_amination/aryl_halide_mordred.csv')
    elif aryl_halide == 'ohe':
        aryl_features = pd.read_csv('data/aryl_amination/aryl_halide_ohe.csv')
        
    # Additive features
    
    if additive == 'dft':
        add_features = pd.read_csv('data/aryl_amination/additive_dft.csv')
    elif additive == 'mordred':
        add_features = pd.read_csv('data/aryl_amination/additive_mordred.csv')
    elif additive == 'ohe':
        add_features = pd.read_csv('data/aryl_amination/additive_ohe.csv')
        
    # Base features
    
    if base == 'dft':    
        base_features = pd.read_csv('data/aryl_amination/base_dft.csv')
    elif base == 'mordred':
        base_features = pd.read_csv('data/aryl_amination/base_mordred.csv')
    elif base == 'ohe':
        base_features = pd.read_csv('data/aryl_amination/base_ohe.csv')
        
    # Ligand features    
    
    if ligand == 'Pd(0)-dft':
        ligand_features = pd.read_csv('data/aryl_amination/ligand-Pd(0)_dft.csv')        
    elif ligand == 'mordred':       
        ligand_features = pd.read_csv('data/aryl_amination/ligand_mordred.csv')
    elif ligand == 'ohe':
        ligand_features = pd.read_csv('data/aryl_amination/ligand_ohe.csv')
        
    # Build the descriptor set
    
    index_list = [index['Aryl_halide_SMILES'],
                  index['Additive_SMILES'],
                  index['Base_SMILES'],
                  index['Ligand_SMILES']]
    
    lookup_table_list = [aryl_features, 
                         add_features, 
                         base_features, 
                         ligand_features]
    
    lookup_list = ['aryl_halide_SMILES', 
                   'additive_SMILES', 
                   'base_SMILES', 
                   'ligand_SMILES']

    experiment_index = build_experiment_index(index['entry'], 
                                              index_list, 
                                              lookup_table_list,
                                              lookup_list)

    experiment_index['yield'] = index['yield'].values
    
    return experiment_index

def suzuki(electrophile='ohe', nucleophile='ohe', base='ohe', ligand='ohe', solvent='ohe'):
    """
    Load Suzuki data with different features.
    """
    
    # SMILES index
    
    index = pd.read_csv('data/suzuki/experiment_index.csv')
    
    # Electrophile features
    
    if electrophile == 'dft':
        elec_features = pd.read_csv('data/suzuki/electrophile_dft.csv')
    elif electrophile == 'mordred':
        elec_features = pd.read_csv('data/suzuki/electrophile_mordred.csv')
    elif electrophile == 'ohe':
        elec_features = pd.read_csv('data/suzuki/electrophile_ohe.csv')
        
    # Nucleophile features
    
    if nucleophile == 'dft':
        nuc_features = pd.read_csv('data/suzuki/nucleophile_dft.csv')
    elif nucleophile == 'mordred':
        nuc_features = pd.read_csv('data/suzuki/nucleophile_mordred.csv')
    elif nucleophile == 'ohe':
        nuc_features = pd.read_csv('data/suzuki/nucleophile_ohe.csv')
        
    # Base features
    
    if base == 'dft':    
        base_features = pd.read_csv('data/suzuki/base_dft.csv')
    elif base == 'mordred':
        base_features = pd.read_csv('data/suzuki/base_mordred.csv')
    elif base == 'ohe':
        base_features = pd.read_csv('data/suzuki/base_ohe.csv')
        
    # Ligand features    
          
    if ligand == 'random-dft':
        ligand_features = pd.read_csv('data/suzuki/ligand-random_dft.csv')   
    elif ligand == 'boltzmann-dft':
        ligand_features = pd.read_csv('data/suzuki/ligand-boltzmann_dft.csv')
    elif ligand == 'mordred':       
        ligand_features = pd.read_csv('data/suzuki/ligand_mordred.csv')
    elif ligand == 'ohe':
        ligand_features = pd.read_csv('data/suzuki/ligand_ohe.csv')
        
    # Solvent features
    
    if solvent == 'dft':
        solvent_features = pd.read_csv('data/suzuki/solvent_dft.csv')
    elif solvent == 'mordred':
        solvent_features = pd.read_csv('data/suzuki/solvent_mordred.csv')
    elif solvent == 'ohe':
        solvent_features = pd.read_csv('data/suzuki/solvent_ohe.csv')
        
    # Build the descriptor set
    
    index_list = [index['Electrophile_SMILES'],
                  index['Nucleophile_SMILES'],
                  index['Base_SMILES'],
                  index['Ligand_SMILES'],
                  index['Solvent_SMILES']]
    
    lookup_table_list = [elec_features, 
                         nuc_features, 
                         base_features, 
                         ligand_features,
                         solvent_features]
    
    lookup_list = ['electrophile_SMILES', 
                   'nucleophile_SMILES', 
                   'base_SMILES', 
                   'ligand_SMILES',
                   'solvent_SMILES']

    experiment_index = build_experiment_index(index['entry'], 
                                              index_list, 
                                              lookup_table_list,
                                              lookup_list)

    experiment_index['yield'] = index['yield']
    
    return experiment_index

def direct_arylation(base='ohe', ligand='ohe', solvent='ohe'):
    """
    Load direct arylation data with different features.
    """
    
    # SMILES index
    index = pd.read_csv('data/direct_arylation/experiment_index.csv')
    
    # Base features
    
    if base == 'dft':    
        base_features = pd.read_csv('data/direct_arylation/base_dft.csv')
    elif base == 'mordred':
        base_features = pd.read_csv('data/direct_arylation/base_mordred.csv')
    elif base == 'ohe':
        base_features = pd.read_csv('data/direct_arylation/base_ohe.csv')
        
    # Ligand features    
          
    if ligand == 'random-dft':
        ligand_features = pd.read_csv('data/direct_arylation/ligand-random_dft.csv')   
    elif ligand == 'boltzmann-dft':
        ligand_features = pd.read_csv('data/direct_arylation/ligand-boltzmann_dft.csv')
    elif ligand == 'mordred':       
        ligand_features = pd.read_csv('data/direct_arylation/ligand_mordred.csv')
    elif ligand == 'ohe':
        ligand_features = pd.read_csv('data/direct_arylation/ligand_ohe.csv')
        
    # Solvent features
    
    if solvent == 'dft':
        solvent_features = pd.read_csv('data/direct_arylation/solvent_dft.csv')
    elif solvent == 'mordred':
        solvent_features = pd.read_csv('data/direct_arylation/solvent_mordred.csv')
    elif solvent == 'ohe':
        solvent_features = pd.read_csv('data/direct_arylation/solvent_ohe.csv')
        
    # Build the descriptor set
    
    index_list = [index['Base_SMILES'],
                  index['Ligand_SMILES'],
                  index['Solvent_SMILES']]
    
    lookup_table_list = [base_features, 
                         ligand_features,
                         solvent_features]
    
    lookup_list = ['base_SMILES', 
                   'ligand_SMILES',
                   'solvent_SMILES']

    experiment_index = build_experiment_index(index['entry'], 
                                              index_list, 
                                              lookup_table_list,
                                              lookup_list)

    experiment_index['concentration'] = index['Concentration']
    experiment_index['temperature'] = index['Temp_C']
    experiment_index['yield'] = index['yield']
    
    return experiment_index


def direct_arylation_extended(base='ohe', ligand='ohe', solvent='ohe'):
    """
    Load direct arylation data for extended ligand set with 
    different features.
    """
    
    # SMILES index
    index = pd.read_csv('data/direct_arylation_extended/experiment_index.csv')
    
    # Base features
    
    if base == 'dft':    
        base_features = pd.read_csv('data/direct_arylation_extended/base_dft.csv')
    elif base == 'mordred':
        base_features = pd.read_csv('data/direct_arylation_extended/base_mordred.csv')
    elif base == 'ohe':
        base_features = pd.read_csv('data/direct_arylation_extended/base_ohe.csv')
        
    # Ligand features    
          
    if ligand == 'random-dft':
        ligand_features = pd.read_csv('data/direct_arylation_extended/ligand-random_dft.csv')   
    elif ligand == 'boltzmann-dft':
        ligand_features = pd.read_csv('data/direct_arylation_extended/ligand-boltzmann_dft.csv')
    elif ligand == 'mordred':       
        ligand_features = pd.read_csv('data/direct_arylation_extended/ligand_mordred.csv')
    elif ligand == 'ohe':
        ligand_features = pd.read_csv('data/direct_arylation_extended/ligand_ohe.csv')
        
    # Solvent features
    
    if solvent == 'dft':
        solvent_features = pd.read_csv('data/direct_arylation_extended/solvent_dft.csv')
    elif solvent == 'mordred':
        solvent_features = pd.read_csv('data/direct_arylation_extended/solvent_mordred.csv')
    elif solvent == 'ohe':
        solvent_features = pd.read_csv('data/direct_arylation_extended/solvent_ohe.csv')
        
    # Build the descriptor set
    
    index_list = [index['Base_SMILES'],
                  index['Ligand_SMILES'],
                  index['Solvent_SMILES']]
    
    lookup_table_list = [base_features, 
                         ligand_features,
                         solvent_features]
    
    lookup_list = ['base_SMILES', 
                   'ligand_SMILES',
                   'solvent_SMILES']

    experiment_index = build_experiment_index(index['entry'], 
                                              index_list, 
                                              lookup_table_list,
                                              lookup_list)

    experiment_index['concentration'] = index['Concentration']
    experiment_index['temperature'] = index['Temp_C']
    experiment_index['yield'] = index['yield']
    
    return experiment_index
