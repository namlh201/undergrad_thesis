import torch

from rdkit.Chem import Mol
from rdkit.Chem.rdmolops import Get3DDistanceMatrix
from rdkit.Chem.rdMolTransforms import GetAngleRad

from .utils.values import VALUES

def featurize_atom(mol: Mol) -> dict[str, torch.Tensor]:
    def _get_atom_type_feature(mol) -> torch.Tensor:
        # num_As x 119
        nrows = mol.GetNumAtoms()
        ncols = VALUES.get_atom_feat_size('atom_type')

        atom_type = torch.zeros((nrows, ncols))

        for i in range(nrows):
            atom_type[i][mol.GetAtomWithIdx(i).GetAtomicNum()] = 1

        return atom_type

    def _get_is_aromatic_feature(mol) -> torch.Tensor:
        # num_As x 2
        nrows = mol.GetNumAtoms()
        ncols = VALUES.get_atom_feat_size('is_aromatic')

        is_aromatic = torch.zeros((nrows, ncols))

        for i in range(nrows):
            if mol.GetAtomWithIdx(i).GetIsAromatic():
                is_aromatic[i][1] = 1
            else:
                is_aromatic[i][0] = 1

        return is_aromatic

    def _get_formal_charge_feature(mol) -> torch.Tensor:
        # num_As x 16
        nrows = mol.GetNumAtoms()
        ncols = VALUES.get_atom_feat_size('formal_charge') # range(-5, 11)

        formal_charge = torch.zeros((nrows, ncols))

        for i in range(nrows):
            formal_charge[i][mol.GetAtomWithIdx(i).GetFormalCharge() + 5] = 1

        return formal_charge

    def _get_chiral_tag_feature(mol) -> torch.Tensor:
        # CHI_UNSPECIFIED       = 0
        # CHI_TETRAHEDRAL_CW    = 1
        # CHI_TETRAHEDRAL_CCW   = 2
        # CHI_OTHER             = 3

        # num_As x 4
        nrows = mol.GetNumAtoms()
        ncols = VALUES.get_atom_feat_size('chiral_tag') # 4

        chiral_tag = torch.zeros((nrows, ncols))

        for i in range(nrows):
            chiral_tag[i][mol.GetAtomWithIdx(i).GetChiralTag()] = 1

        return chiral_tag
    
    def _get_degree_feature(mol) -> torch.Tensor:
        # num_As x 11
        nrows = mol.GetNumAtoms()
        ncols = VALUES.get_atom_feat_size('degree')

        degree = torch.zeros((nrows, ncols))

        for i in range(nrows):
            degree[i][mol.GetAtomWithIdx(i).GetTotalDegree()] = 1

        return degree

    def _get_num_bonded_Hs_feature(mol) -> torch.Tensor:
        # num_As x 9
        nrows = mol.GetNumAtoms()
        ncols = VALUES.get_atom_feat_size('num_bonded_Hs')

        num_bonded_Hs = torch.zeros((nrows, ncols))

        for i in range(nrows):
            num_bonded_Hs[i][mol.GetAtomWithIdx(i).GetTotalNumHs(includeNeighbors=True)] = 1

        return num_bonded_Hs

    def _get_hybridization_feature(mol) -> torch.Tensor:
        # SP    = 2
        # SP2   = 3
        # SP3   = 4
        # SP3D  = 5
        # SP3D2 = 6

        # num_As x 8
        nrows = mol.GetNumAtoms()
        ncols = VALUES.get_atom_feat_size('hybridization')

        hybridization = torch.zeros((nrows, ncols))

        for i in range(nrows):
            hybridization[i][mol.GetAtomWithIdx(i).GetHybridization()] = 1

        return hybridization

    atom_features = {
        'atom_type': _get_atom_type_feature(mol),
        'is_aromatic': _get_is_aromatic_feature(mol),
        'formal_charge': _get_formal_charge_feature(mol),
        'chiral_tag': _get_chiral_tag_feature(mol),
        'degree': _get_degree_feature(mol),
        'num_bonded_Hs': _get_num_bonded_Hs_feature(mol),
        'hybridization': _get_hybridization_feature(mol)
    }

    return atom_features

def featurize_atom_mass(mol: Mol) -> dict[str, torch.Tensor]:
    def _get_atom_mass_feature(mol) -> torch.Tensor:
        # num_As x 1
        nrows = mol.GetNumAtoms()
        
        atom_mass = torch.zeros((nrows,))

        # dist_mat = Get3DDistanceMatrix(mol)

        for i in range(nrows):
            atom_mass[i] = mol.GetAtomWithIdx(i).GetMass()

        return atom_mass

    features = {
        'mass': _get_atom_mass_feature(mol)
    }

    return features

def featurize_bond(mol: Mol) -> dict[str, torch.Tensor]:
    def _get_bond_dir_feature(mol) -> torch.Tensor:
        # NONE          = 0
        # BEGINWEDGE    = 1
        # BEGINDASH     = 2
        # ENDDOWNRIGHT  = 3
        # ENDUPRIGHT    = 4
        # EITHERDOUBLE  = 5
        # UNKNOWN       = 6

        # num_Bs x 7
        nrows = mol.GetNumBonds()
        ncols = VALUES.get_bond_feat_size('bond_dir') # 7

        bond_dir = torch.zeros((nrows, ncols))

        for i in range(nrows):
            bond_dir[i][mol.GetBondWithIdx(i).GetBondDir()] = 1
        
        return bond_dir

    def _get_bond_type_feature(mol) -> torch.Tensor:
        # SINGLE    = 1 -> 0
        # DOUBLE    = 2 -> 1
        # TRIPLE    = 3 -> 2
        # AROMATIC = 12 -> 3
        
        # value_to_idx = {
        #     BondType.SINGLE:    0,
        #     BondType.DOUBLE:    1,
        #     BondType.TRIPLE:    2,
        #     BondType.AROMATIC:  3
        # }

        # num_Bs x 13
        nrows = mol.GetNumBonds()
        ncols = VALUES.get_bond_feat_size('bond_type')

        bond_type = torch.zeros((nrows, ncols))

        for i in range(nrows):
            bond_type[i][mol.GetBondWithIdx(i).GetBondType()] = 1
        
        return bond_type

    def _get_is_in_ring_feature(mol) -> torch.Tensor:
        # num_Bs x 2
        nrows = mol.GetNumBonds()
        ncols = VALUES.get_bond_feat_size('is_in_ring')

        in_ring = torch.zeros((nrows, ncols))

        for i in range(nrows):
            if mol.GetBondWithIdx(i).IsInRing():
                in_ring[i][1] = 1
            else:
                in_ring[i][0] = 1
        
        return in_ring

    def _get_is_conjugated_feature(mol) -> torch.Tensor:
        # num_Bs x 2
        nrows = mol.GetNumBonds()
        ncols = VALUES.get_bond_feat_size('is_conjugated')

        conjugated = torch.zeros((nrows, ncols))

        for i in range(nrows):
            if mol.GetBondWithIdx(i).GetIsConjugated():
                conjugated[i][1] = 1
            else:
                conjugated[i][0] = 1
        
        return conjugated

    bond_features = {
        'bond_dir': _get_bond_dir_feature(mol),
        'bond_type': _get_bond_type_feature(mol),
        'is_in_ring': _get_is_in_ring_feature(mol),
        'is_conjugated': _get_is_conjugated_feature(mol)
    }

    return bond_features

def featurize_bond_length(mol: Mol) -> dict[str, torch.Tensor]:
    def _get_bond_length_feature(mol) -> torch.Tensor:
        # num_Bs x 1
        bonds = mol.GetBonds()
        nrows = len(bonds)
        # nrows = mol.GetNumBonds(onlyHeavy=False)
        # print(nrows)

        bond_length = torch.zeros((nrows,))

        dist_mat = Get3DDistanceMatrix(mol)

        for i in range(nrows):
            begin_atom_idx = mol.GetBondWithIdx(i)\
                .GetBeginAtomIdx()
            end_atom_idx = mol.GetBondWithIdx(i).GetEndAtomIdx()

            bond_length[i] = dist_mat[begin_atom_idx][end_atom_idx]

        return bond_length

    features = {
        'length': _get_bond_length_feature(mol)
    }

    return features

def featurize_bond_angle(mol) -> dict[str, torch.Tensor]:
    def _get_bond_angles(mol) -> list:
        # Return a list of bond-angles, with each entry as an indice tuple of (end_atom_1, connecting_atom, end_atom_2)
        bonds = mol.GetBonds()
        # atom_indices = map(list, [atom.GetIdx() for atom in mol.GetAtoms()])
        num_bonds = len(bonds)

        bond_couples = []
        
        bond_angles = []

        for i in range(num_bonds):
            for j in range(num_bonds):
                if i == j: continue

                if (i, j) in bond_couples or (j, i) in bond_couples: continue
        
                if bonds[i].GetBeginAtomIdx() == bonds[j].GetBeginAtomIdx():
                    bond_angles.append((
                        bonds[i].GetEndAtomIdx(),
                        bonds[i].GetBeginAtomIdx(),
                        bonds[j].GetEndAtomIdx()
                    ))

                elif bonds[i].GetBeginAtomIdx() == bonds[j].GetEndAtomIdx():
                    bond_angles.append((
                        bonds[i].GetEndAtomIdx(),
                        bonds[i].GetBeginAtomIdx(),
                        bonds[j].GetBeginAtomIdx()
                    ))

                elif bonds[i].GetEndAtomIdx() == bonds[j].GetBeginAtomIdx():
                    bond_angles.append((
                        bonds[i].GetBeginAtomIdx(),
                        bonds[i].GetEndAtomIdx(),
                        bonds[j].GetEndAtomIdx()
                    ))

                elif bonds[i].GetEndAtomIdx() == bonds[j].GetEndAtomIdx():
                    bond_angles.append((
                        bonds[i].GetBeginAtomIdx(),
                        bonds[i].GetEndAtomIdx(),
                        bonds[j].GetBeginAtomIdx()
                    ))

                bond_couples.append((i, j))
                bond_couples.append((j, i))

        # bond_angles = list(set(bond_angles))
        # print(bond_angles)
        
        return bond_angles

    def _get_bond_angle_feature(mol: Mol) -> torch.Tensor:
        # num_BAs x 1
        conf = mol.GetConformer()
        bond_angles_list = _get_bond_angles(mol)

        # bond_angle_degree = [GetAngleDeg(conf, ba[0], ba[1], ba[2]) for ba in bond_angles_list]
        # print(bond_angle_degree)
        # print(len(bond_angle_degree))
        # bond_angle_degree = torch.tensor(bond_angle_degree)
        # bond_angle_radian = bond_angle_degree.deg2rad()

        bond_angle_radian = [GetAngleRad(conf, ba[0], ba[1], ba[2]) for ba in bond_angles_list]
        bond_angle_radian = torch.tensor(bond_angle_radian)

        return bond_angle_radian

    features = {
        'angle': _get_bond_angle_feature(mol)
    }

    return features