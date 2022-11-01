import numpy
import math, os, time, sys
import h5py
import scipy.sparse
from pyscf import gto, scf, mcscf, fci, ao2mo, lib, tools, cc
import scipy.linalg as la
from pyscf.shciscf import shci


# Cu2O2^2+: bis f = 0., per f = 1.
if len(sys.argv) == 2:
    f = float(sys.argv[1])
else:
    f = 0.

# bond lengths
cu_cu = 2.8 + f * 0.8
o_o = 2.3 - f * 0.9

atomString = f'Cu {-cu_cu/2} 0. 0.; Cu {cu_cu/2} 0. 0.; O 0. {o_o/2} 0.; O 0. {-o_o/2} 0.'

mol1 = gto.M(atom = atomString, basis = {'Cu': 'ano@5s4p2d1f', 'O': 'ano@3s2p1d'},
    verbose = 4, unit = 'angstrom', symmetry = 1, spin = 0, charge = 2)
mol = gto.M(atom = atomString, basis = {'Cu': 'ano@6s5p3d2f1g', 'O': 'ano@4s3p2d1f'},
    verbose = 4, unit = 'angstrom', symmetry = 1, spin = 0, charge = 2)
mf = scf.RHF(mol).x2c()
#mf.chkfile = "/burg/ccce/users/jl5653/cu2o2/ANODZ/44e/f_0.0/f_0.0_natorb_shciscf/cu2o2_0.0_SHCISCF.chk"
mf.chkfile = f"mf.chk"
#mf.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
#mf.init_guess = '1e'
mf.level_shift = 0.1
mf.scf()
print(f'mol.nao: {mol.nao}')
print(f'mol.elec: {mol.nelec}')

ncore = 14 + 6
nfzv = 44
norbAct = mol1.nao_nr() - ncore - nfzv

mc = shci.SHCISCF(mf, norbAct, mol.nelectron - 2*ncore)
#mo_init = lib.chkfile.load(f'cu2o2_{f}_SHCISCF.chk', 'mcscf/mo_coeff')
#mc.__dict__.update(lib.chkfile.load(chkfile, 'mcscf'))
#mc.chkfile = f'cu2o2_{f}_SHCISCF.chk'
mc.fcisolver.sweep_iter = [ 0, 5,10 ]
mc.fcisolver.sweep_epsilon = [ 1e-3, 5e-4, 1.0e-4 ]
mc.fcisolver.nPTiter = 0
mc.fcisolver.stochastic = True
mc.max_cycle_macro = 20
mc.internal_rotation = True
mc.fcisolver.nPTiter = 0  # Turns off PT calculation, i.e. no PTRDM.
mc.fcisolver.mpiprefix = "mpirun -np 32"
#mc.fcisolver.scratchDirectory = "/rc_scratch/anma2640/cu2o2/1.0"

#mo_init_guess = lib.chkfile.load("/burg/ccce/users/jl5653/cu2o2/ANODZ/44e/f_1.0/f_1.0_natorb_shciscf/cu2o2.chk", 'mcscf/mo_coeff')
#f = open("/burg/ccce/users/jl5653/cu2o2/ANODZ/44e/f_1.0/f_1.0_natorb_shciscf/spatial1RDM.0.0.txt", "r")
#lines = f.readlines()
#f.close()
#norb = int(lines[0].strip())
#
#P = numpy.zeros((norb, norb))
#for line in lines[1:]:
#    line = line.strip().split()
#    i = int(line[0])
#    j = int(line[1])
#    val = float(line[2])
#    P[i,j] = val
#
#e, v = numpy.linalg.eigh(P)
#v = v[:,::-1]
#e = e[::-1]
#Cact = mo_init_guess[:,ncore:ncore+norbAct].dot(v)
#mo_init_guess[:,ncore:ncore+norbAct] = Cact.copy()
#
#mo = mcscf.project_init_guess(mc, mo_init_guess, mol1)

mo = lib.chkfile.load(f'cu2o2_{f}_SHCISCF.chk', 'mcscf/mo_coeff')
mc.chkfile = f'cu2o2_{f}_SHCISCF.chk'
e_noPT = mc.mc1step(mo_coeff=mo)[0]

### Run a single SHCI iteration with perturbative correction.
#mc.fcisolver.stochastic = True  # Turns on deterministic PT calc.
#mc.fcisolver.epsilon2 = 1e-7
#mc.fcisolver.nPTiter = 20  # Turn off perturbative calc.
#shci.writeSHCIConfFile(mc.fcisolver, [mol.nelec[0] - 2, mol.nelec[1]-2], False)
#shci.executeSHCI(mc.fcisolver)
#e_PT = shci.readEnergy(mc.fcisolver)
##
## Comparison Calculations
#del_PT = e_PT - e_noPT
#
#print("del_PT = {}".format(del_PT))
#
#mycas = mcscf.CASSCF(mf, ncas, nelecas)
#mycas.mo_coeff = mc.mo_coeff
#print("mc.mo_coeff.shape = {}".format(mc.mo_coeff.shape))
#h1e_cas, ecore = mycas.get_h1eff()
#h2e_cas = mycas.get_h2eff()
#tools.fcidump.from_integrals('FCIDUMP_full', h1e_cas, h2e_cas, h1e_cas.shape[-1],
#                             nelecas, nuc = ecore, ms=0, tol=1e-8)
