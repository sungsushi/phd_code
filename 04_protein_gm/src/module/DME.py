"""
module adapted from pdb_chop by runfeng
"""
from Bio.PDB import PDBParser, PDBIO, Dice
import sys
import os
import subprocess as sp


# domain match evaluation using TMalign
class DMatchEval:
    def __init__(self, 
                 contact_map_path=None, 
                 pdb_path=None, 
                 domain_path=None, 
                 tm_align=None): 
        if contact_map_path==None:
            contact_map_path = '/Users/ssm47/Library/CloudStorage/OneDrive-UniversityofCambridge/ssnw' # this is the path to your contact map folder
        if pdb_path==None:
            pdb_path = '/Users/ssm47/Documents/contact_maps/data/pdbfiles/database' # this is the dirctory to your pdb folder
        if domain_path==None:
            domain_path = '/Users/ssm47/Documents/contact_maps/data/pdbfiles/chopped' # this is the path to save your chopped pdb file
        if tm_align==None:
            tm_align = 'TMalign' # path to your TMalign 

        self.contact_map_path = contact_map_path
        self.pdb_path = pdb_path
        self.domain_path = domain_path
        self.tm_align = tm_align
    
    def ss_extraction(self,protein):
        f = open(f'{self.contact_map_path}/{protein}.ssnw')
        length = f.readline().strip().split()[1:]
        element = f.readline().strip().split()[1:]
        # print(element,length)
        ss_dict = {}
        ss_i_dict = {}
        start = 0
        ss = []
        for i in range(len(element)):
            if 'T' not in element[i]:
                ss_dict[(element[i][0],int(length[i]),start)]=len(ss_dict)
                ss_i_dict[len(ss_i_dict)] = (element[i][0],int(length[i]),start)
                ss.append((element[i][0],int(length[i]),start))
            start += int(length[i])
        return ss

    def map_seq_index_to_resid(self, structure, chain_id):
        chain = structure[0][chain_id]  
        seq_to_resid = {}
        for i, residue in enumerate(chain.get_residues(), start=1):
            seq_to_resid[i] = residue.id[1]  
        return seq_to_resid


    def prep_pdb_chop(self, protein, idxX0, idxX1):
        ss = self.ss_extraction(protein)
        idxX0 = ss[idxX0][2]
        idxX1 = ss[idxX1][2]+ss[idxX1][1]
        # idxX1 = ss[idxX1+1][2] if idxX1<len(ss)-1 else ss[idxX1][2]+ss[idxX1][1]
        parser = PDBParser()
        # structure = parser.get_structure(protein, f'{self.pdb_path}/{protein[1:3]}/{protein[:4]}.pdb')
        structure = parser.get_structure(protein, f'{self.pdb_path}/{protein[:4]}.pdb')
        seq_to_residX = self.map_seq_index_to_resid(structure, protein[-1]) 
        residX0 = seq_to_residX[idxX0]
        residX1 = seq_to_residX[idxX1]
        domain_filename = f'{self.domain_path}/{protein}.pdb'
        Dice.extract(structure, protein[-1], residX0, residX1, domain_filename)
        # print(f'{protein} chopped between indices {idxX0} - {idxX1} saved at \n domain_filename')

    def tma(self, protein1, protein2):
        '''Performs the tmalign and outputs the readout of scores and alignment'''
        aligned_residues = sp.getoutput(f'{self.tm_align} {self.domain_path}/{protein1}.pdb {self.domain_path}/{protein2}.pdb')
        align1 = aligned_residues[aligned_residues.index('other aligned residues)')+24:].strip().split('\n')
        ar=aligned_residues.index("TM-score=")
        # br=aligned_residues.index('(if normalized by length of Chain_2)') 
        br=aligned_residues.index('(normalized by length of Structure_2:') 
        query_score = float(aligned_residues[ar+10:ar+18])
        target_score = float(aligned_residues[br-8:br])    
        ms = (query_score+target_score)/2

        structural_alignment = f'\n{align1[0]}\n{align1[1]}\n{align1[2]}'
        output = {'query_score':query_score, 'target_score':target_score,\
                   'ms':ms, 'structural_alignment':structural_alignment}
        return output

    def match_score(self, protein1, protein2, idx0_range, idx1_range, printit=False):
        '''
        Perform the tmalign of two proteins with input index ranges of the matching.

        idx0_range : ss_element indices of protein 1 (coil ignored)
        idx1_range : ss_element indices of protein 2 (coil ignored)
        '''
        idx00 = idx0_range[0]
        idx01 = idx0_range[1]
        idx10 = idx1_range[0]
        idx11 = idx1_range[1]

        # saves chopped pdb files:
        self.prep_pdb_chop(protein=protein1, idxX0=idx00, idxX1=idx01) 
        self.prep_pdb_chop(protein=protein2, idxX0=idx10, idxX1=idx11)

        readout = self.tma(protein1=protein1, protein2=protein2) # performs tmalign
        query_score = readout.get('query_score')
        target_score = readout.get('target_score')
        ms = readout.get('ms')
        structural_alignment = readout.get('structural_alignment')
        if printit:
            print(f'{protein1} TM score:{query_score:.3f}, {protein2} TM score:{target_score:.3f}, mean TM score: {ms:.3f}\n Structual alignments:{structural_alignment}')
        return readout


if __name__ == "__main__":

    protein1 = '5y3n_A' #'3aig_A'
    idx0_range = [0,33] #[0,8]
    protein2 = '4z1f_A' #'3k7n_A'
    idx1_range =  [1,35] #[0,8]

    dme = DMatchEval()

    readout = dme.match_score(protein1=protein1, 
                              protein2=protein2, 
                              idx0_range=idx0_range, 
                              idx1_range=idx1_range, 
                              printit=True)
    















