from rdkit.Chem.rdchem import BondDir, BondType, ChiralType, HybridizationType

class VALUES():
    atom_feat_values = {
        'atom_type': list(range(0, 119)),
        'is_aromatic': [0, 1],
        'formal_charge': list(range(0, 16)), # range(-5, 11)
        'chiral_tag': list(map(int, ChiralType.values)),
        'degree': list(range(0, 11)),
        'num_bonded_Hs': list(range(0, 9)),
        'hybridization': list(map(int, HybridizationType.values))
    }

    bond_feat_values = {
        'bond_dir': list(map(int, BondDir.values)),
        'bond_type': list(map(int, BondType.values)),
        'is_in_ring': [0, 1],
        'is_conjugated': [0, 1]
    }

    # bond_length_feat_values = {
    #     'length': 
    # }
    @staticmethod
    def get_atom_feat_names() -> list:
        return list(VALUES.atom_feat_values.keys())

    @staticmethod
    def get_atom_feat_vals(name: str) -> list:
        return VALUES.atom_feat_values[name]

    @staticmethod
    def get_atom_feat_size(name: str) -> int:
        return len(VALUES.atom_feat_values[name])

    @staticmethod
    def get_bond_feat_names() -> list:
        return list(VALUES.bond_feat_values.keys())
    
    @staticmethod
    def get_bond_feat_vals(name: str) -> list:
        return VALUES.bond_feat_values[name]

    @staticmethod
    def get_bond_feat_size(name: str) -> int:
        return len(VALUES.bond_feat_values[name])