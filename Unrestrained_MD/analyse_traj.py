import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis.distances import dist
import pickle
import sys
from tqdm import tqdm
import os

def getDistance(idx1, idx2, u):
    """
    Get the distance between two atoms in a universe.

    Parameters
    ----------
    idx1 : int
        Index of the first atom
    idx2 : int
        Index of the second atom
    u : MDAnalysis.Universe
        The MDA universe containing the atoms and
        trajectory.

    Returns
    -------
    distance : float
        The distance between the two atoms in Angstroms.
    """
    distance = dist(
        mda.AtomGroup([u.atoms[idx1]]),
        mda.AtomGroup([u.atoms[idx2]]),
        box=u.dimensions,
    )[2][0]
    return distance

def closest_residue_to_point(atoms, point):
    """Find the closest residue in a selection of atoms to a given point"""
    residues = atoms.residues
    distances = np.array([np.linalg.norm(res.atoms.center_of_mass() - point) for res in residues])

    # Find the index of the smallest distance
    closest_residue_index = np.argmin(distances)

    # Return the closest residue
    return residues[closest_residue_index], distances[closest_residue_index]

def obtain_CA_idx(u, res_idx):
    """Function to obtain the index of the alpha carbon for a given residue index"""
    
    selection_str = f"protein and resid {res_idx} and name CA"
    
    selected_CA = u.select_atoms(selection_str)

    if len(selected_CA.indices) == 0:
        print('CA not found for the specified residue...')
    
    elif len(selected_CA.indices) > 1:
        print('Multiple CAs found, uh oh...')

    else:  
        return selected_CA.indices[0]
    
def obtain_angle(run_number, pos1, pos2, pos3):

    u = mda.Universe('structures/complex.prmtop', f'results/run{run_number}/traj.dcd')

    return mda.lib.distances.calc_angles(pos1, pos2, pos3)

def obtain_dihedral(run_number, pos1, pos2, pos3, pos4):
    
    u = mda.Universe('structures/complex.prmtop', f'results/run{run_number}/traj.dcd')

    return mda.lib.distances.calc_dihedrals(pos1, pos2, pos3, pos4)

def obtain_RMSD(run_number, res_range=[0,392]):
    u = mda.Universe('structures/complex.prmtop', f'results/run{run_number}/traj.dcd')
    protein = u.select_atoms("protein")

    ref = protein
    R_u =rms.RMSD(protein, ref, select=f'backbone and resid {res_range[0]}-{res_range[1]}')
    R_u.run()

    rmsd_u = R_u.rmsd.T #take transpose
    time = rmsd_u[1]/1000
    rmsd= rmsd_u[2]

    return time, rmsd

def save_RMSD(run_number, res_range=[0,392]):
    """
    Save the RMSD of a given run in a .csv file
    """
    time, RMSD = obtain_RMSD(run_number, res_range)

    df = pd.DataFrame()
    df['Time (ns)'] = time
    df['RMSD (Angstrom)'] = RMSD

    filename = 'RMSD.csv'

    df.to_csv(f"results/run{run_number}/{filename}")

    return df

def obtain_RMSF(run_number, res_range=[0,392]):
    u = mda.Universe('structures/complex.prmtop', f'results/run{run_number}/traj.dcd')
    
    average = align.AverageStructure(u, u, select='protein and name CA',
                                 ref_frame=0).run()
    
    ref = average.results.universe
    
    average = align.AverageStructure(u, u, select='protein and name CA', ref_frame=0).run()

    aligner = align.AlignTraj(u, ref,
                            select='protein and name CA',
                            in_memory=True).run()

    c_alphas = u.select_atoms(f'protein and name CA and resid {res_range[0]}-{res_range[1]}')
    R = rms.RMSF(c_alphas).run()

    res = c_alphas.resids
    rmsf = R.results.rmsf

    return res, rmsf

def save_RMSF(run_number, res_range=[0,392]):
    """
    Save the RMSD of a given run in a .csv file
    """
    residx, RMSF = obtain_RMSF(run_number, res_range)

    df = pd.DataFrame()
    df['Residue index'] = residx
    df['RMSF (Angstrom)'] = RMSF
    df.to_csv(f"results/run{run_number}/RMSF.csv")

    return df

def run_analysis(systems, k_values):

    for system in systems:
        for k_DDB1 in k_values:
            for n_run in [1,2,3]:
                print(f"\nGenerating RMSD 1for {system}, run {n_run}, with k={k_DDB1} kcal/mol AA^-2\n")
                save_RMSD(system, k_DDB1, n_run, glob=True)
                save_RMSD(system, k_DDB1, n_run, glob=False)
                print(f"\nGenerating RMSF for {system}, run {n_run}, with k={k_DDB1} kcal/mol AA^-2\n")
                save_RMSF(system, k_DDB1, n_run)

def obtain_Boresch_dof(run_number, dof):

    rec_group = [4, 18, 37, 56, 81, 96, 107, 126, 136, 160, 177, 193, 215, 226, 245, 264, 286, 307, 318, 332, 346, 400, 406, 425, 447, 453, 521, 610, 629, 649, 655, 666, 688, 694, 710, 727, 789, 941, 1872, 1899, 1905, 1920, 1941, 1960, 1999, 2026, 2057, 2068, 2084, 2095, 2102, 2112, 2123, 2133, 2140, 2164, 2183, 2197, 2219, 2463]
    lig_group = [3032, 3054, 3071, 3088, 3333, 3360, 3366, 3378, 3442, 4051, 4072, 4091, 4112, 4126, 4156, 4162, 4169, 4181, 4193, 4212, 4228, 4247, 4264, 4274, 4289, 4824, 4834, 4856, 4878, 4895, 4905, 4915, 4936, 4946, 4978, 5025, 5055, 5061, 5077, 5089, 5105, 5120, 5130, 5149, 5156, 5175, 5192, 5204, 5890, 5911, 5933, 5953, 5965, 5982, 5997, 6013, 6039]

    res_b = 8
    res_c = 142
    res_B = 332
    res_C = 244

    group_a = u.atoms[rec_group]
    group_b = u.atoms[[obtain_CA_idx(u, res_b)]]
    group_c = u.atoms[[obtain_CA_idx(u, res_c)]]
    group_A = u.atoms[lig_group]
    group_B = u.atoms[[obtain_CA_idx(u, res_B)]]
    group_C = u.atoms[[obtain_CA_idx(u, res_C)]]

    pos_a = group_a.center_of_mass()
    pos_b = group_b.center_of_mass()
    pos_c = group_c.center_of_mass()
    pos_A = group_A.center_of_mass()
    pos_B = group_B.center_of_mass()
    pos_C = group_C.center_of_mass()

    dof_indices = {
        'thetaA' : [pos_b, pos_a, pos_A],
        'thetaB' : [pos_a, pos_A, pos_B],
        'phiA' : [pos_c, pos_b, pos_a, pos_A],
        'phiB': [pos_b, pos_a, pos_A, pos_B],
        'phiC': [pos_a, pos_A, pos_B, pos_C]
    }

    indices = dof_indices[dof]

    if len(indices) == 3:
        return obtain_angle(run_number, indices[0], indices[1], indices[2])

    else:
        return obtain_dihedral(run_number, indices[0], indices[1], indices[2], indices[3])

# for n_run in [0,1,2,3,4]:
#     print(f"\nGenerating RMSD for run {n_run}")
#     save_RMSD(n_run)
#     print(f"\nGenerating RMSF for  run {n_run}")
#     save_RMSF( n_run)   

dof = str(sys.argv[1])

for run_number in [4]:

    # for dof in ['thetaA', 'thetaB', 'phiA', 'phiB', 'phiC']:
    if os.path.exists(f'results/run{run_number}/{dof}.pkl'):
        continue
    else:
        print(f"Performing Boresch analysis for {dof} run {run_number}")

        u = mda.Universe('structures/complex.prmtop', f'results/run{run_number}/traj.dcd')

        vals = []

        for ts in tqdm(u.trajectory, total=u.trajectory.n_frames, desc='Frames analysed'):
            vals.append(obtain_Boresch_dof(run_number, dof))

        frames = np.arange(1, len(vals) + 1)

        dof_data = {
            'Frames': frames,
            'Time (ns)': np.round(0.01 * frames, 6),
            'DOF values': vals
        }

        # Save interface data to pickle
        file = f'results/run{run_number}/{dof}.pkl'
        with open(file, 'wb') as f:
            pickle.dump(dof_data, f)