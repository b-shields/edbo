
from edbo.bro import BO_express
import pandas as pd

ligands = pd.read_csv('ligands.csv').sample(12, random_state=0).values.flatten()
bases = ['DBU', 'MTBD', 'potassium carbonate', 'potassium phosphate', 'potassium tert-butoxide']

reaction_components={'aryl_halide':['chlorobenzene','iodobenzene','bromobenzene'],
                     'base':bases,
                     'solvent':['THF', 'Toluene', 'DMSO', 'DMAc'],
                     'ligand':ligands,
                     'concentration':[0.1, 0.2, 0.3],
                     'temperature': [20, 30, 40]
                     }

encoding={
          'aryl_halide':'resolve',
          'base':'resolve',
          'solvent':'resolve',
          'ligand':'mordred',
          'concentration':'numeric',
          'temperature':'numeric'}

from edbo.utils import timer

t = timer('Init')

bo = BO_express(reaction_components=reaction_components, 
                encoding=encoding,
                batch_size=10,
                acquisition_function='TS',
                target='yield')

t.stop()

#%%

bo.init_sample()
bo.export_proposed()

bo.save()

#%%

from edbo.bro import BO_express

bo = BO_express()
bo.load()

#%%

from edbo.utils import timer

t = timer('Run')

bo.add_results('test_rxn_results.csv')
bo.run()

t.stop()