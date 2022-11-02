import numpy as np
import numpy
import sys
from pyscf import gto, scf, lib
from ipie.utils.from_pyscf import generate_integrals, freeze_core
from pyscf.shciscf import shci


# Cu2O2^2+: bis f = 0., per f = 1.
f = 0.

# bond lengths
cu_cu = 2.8 + f * 0.8
o_o = 2.3 - f * 0.9

atomString = f'Cu {-cu_cu/2} 0. 0.; Cu {cu_cu/2} 0. 0.; O 0. {o_o/2} 0.; O 0. {-o_o/2} 0.'

mol = gto.M(atom = atomString,
    basis = {'O': gto.basis.parse('''
O    S
 105374.95                   0.459000000E-04             -0.105000000E-04            0.900000000E-05             -0.109000000E-04
  15679.240                  0.360700000E-03             -0.825000000E-04            0.705000000E-04             -0.822000000E-04
   3534.5447                 0.191980000E-02             -0.441200000E-03            0.375700000E-03             -0.467500000E-03
    987.36516                0.820670000E-02             -0.188640000E-02            0.161460000E-02             -0.184650000E-02
    315.97875                0.297257000E-01             -0.695400000E-02            0.593400000E-02             -0.755850000E-02
    111.65428                0.904558000E-01             -0.217208000E-01            0.187866000E-01             -0.210868000E-01
     42.699451               0.217405400E+00             -0.568513000E-01            0.494683000E-01             -0.667511000E-01
     17.395596               0.368765700E+00             -0.113963500E+00            0.103039900E+00             -0.109367300E+00
      7.4383090              0.337279800E+00             -0.162020100E+00            0.162058600E+00             -0.273143100E+00
      3.2228620              0.967505000E-01             -0.333800000E-01            0.936700000E-03             0.209713700E+00
      1.2538770              0.256740000E-02             0.365506800E+00             -0.822425100E+00            0.120348070E+01
       .49515500             0.137460000E-02             0.552003100E+00             -0.101790200E+00            -0.677469400E+00
       .19166500             -0.141000000E-03            0.223639300E+00             0.425393900E+00             -0.142988400E+01
       .06708300             0.683000000E-04             0.657450000E-02             0.687702800E+00             0.148910680E+01
O    P
    200.00000                0.893300000E-03            -0.838400000E-03            0.126180000E-02
     46.533367               0.736900000E-02            -0.684910000E-02            0.111628000E-01
     14.621809               0.349392000E-01            -0.328505000E-01            0.518316000E-01
      5.3130640              0.115298500E+00            -0.110006000E+00            0.197884500E+00
      2.1025250              0.258323100E+00            -0.313526300E+00            0.570765200E+00
       .85022300             0.369623100E+00            -0.319601100E+00            -0.178929100E+00
       .33759700             0.323878900E+00            0.221724300E+00             -0.898207700E+00
       .12889200             0.146798900E+00            0.562261600E+00             0.266664300E+00
       .04511200             0.336127000E-01            0.301325100E+00             0.625899400E+00
O    D
      3.7500000              0.12849340            -0.21820550
      1.3125000              0.52118840            -0.48176950
       .45937500             0.43457840             0.13575950
       .16078100             0.14574090             0.82977340
    '''),
    'Cu': gto.basis.parse('''
Cu    S
     27.6963200              0.2311320              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
     13.5053500             -0.6568110              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      8.8153550             -0.5458750              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      2.3808050              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000              0.0000000
      0.9526160              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000              0.0000000
      0.1126620              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000
      0.0404860              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000
      0.0100000              0.0000000              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000
Cu    P
     93.5043270              0.0228290              0.0000000              0.0000000              0.0000000              0.0000000
     16.2854640             -1.0095130              0.0000000              0.0000000              0.0000000              0.0000000
      5.9942360              0.0000000              0.2464500              0.0000000              0.0000000              0.0000000
      2.5368750              0.0000000              0.7920240              0.0000000              0.0000000              0.0000000
      0.8979340              0.0000000              0.0000000              1.0000000              0.0000000              0.0000000
      0.1317290              0.0000000              0.0000000              0.0000000              1.0000000              0.0000000
      0.0308780              0.0000000              0.0000000              0.0000000              0.0000000              1.0000000
Cu    D
     41.2250060              0.0446940              0.0000000              0.0000000
     12.3432500              0.2121060              0.0000000              0.0000000
      4.2019200              0.4534230              0.0000000              0.0000000
      1.3798250              0.5334650              0.0000000              0.0000000
      0.3834530              0.0000000              1.0000000              0.0000000
      0.1000000              0.0000000              0.0000000              1.0000000
    ''')},
    ecp = {'Cu': gto.basis.parse_ecp('''
Cu nelec 10
Cu ul
2       1.000000000            0.000000000
Cu S
2      30.220000000          355.770158000
2      13.190000000           70.865357000
Cu P
2      33.130000000          233.891976000
2      13.220000000           53.947299000
Cu D
2      38.420000000          -31.272165000
2      13.260000000           -2.741104000
    ''')},
    verbose = 4, unit = 'angstrom', symmetry = 0, spin = 0, charge = 2)
mf = scf.RHF(mol)
chkfile = f'cu2o2_32e108o.chk'
# mf.chkfile = f'../cu2o2_{f}_SHCISCF.chk'
mf.__dict__.update(lib.chkfile.load(chkfile, 'scf'))
# mf.init_guess = '1e'
mf.level_shift = 0.1
C = mf.mo_coeff[:,mf.mo_occ>0]
dm0 = 2 * C @ C.T
mf.scf(dm0=dm0)
print(f'mol.nao: {mol.nao}')
print(f'mol.elec: {mol.nelec}')

norbFrozen = 10
ncore = 10
data = lib.chkfile.load(chkfile, 'mcscf')
print(data['e_tot'])
print("nao: ", mol.nao)

def get_opdm(filename):
    with open(filename) as f:
        data = f.readlines()
        nmo = int(data[0])
        opdm = np.zeros((nmo,nmo))
        for l in data[1:]:
            _l = l.split()
            i, j, v = int(_l[0]), int(_l[1]), float(_l[2])
            opdm[i,j] = v
            opdm[j,i] = v

    return opdm

opdm_0 = get_opdm("spatial1RDM.0.0.txt")
no_occ, U_no  = np.linalg.eigh(opdm_0)
no_occ = no_occ[::-1]
U_no = U_no[:,::-1].copy()
keep = np.where(no_occ > 0.01)[0]
nact = len(keep)
print("nact : {}".format(nact))

# transform original SHCISCF orbitals to natural orbital basis
# Only transform non frozen core orbitals.
C_no = data['mo_coeff'].copy()
C_no[:,ncore:] = numpy.dot(data['mo_coeff'][:,ncore:], U_no)

mc1 = shci.SHCISCF(mf,  nact, 32)
# Run SHCISCF within active space with relatively coarse epsilon for obital
# optimization.
mc1.chkfile = f'cu2o2_{f}_SHCISCF.chk'
mc1.frozen = norbFrozen
mc1.fcisolver.sweep_iter = [ 0, 3]
mc1.fcisolver.sweep_epsilon = [ 1e-3, 1e-4 ]
mc1.fcisolver.nPTiter = 0
mc1.max_iter = 10
mc1.fcisolver.maxIter = 20
mc1.max_cycle_macro = 10
mc1.internal_rotation = True
mc1.fcisolver.nPTiter = 0  # Turns off PT calculation, i.e. no PTRDM.
mc1.fcisolver.mpiprefix = "mpirun --map-by ppr:30:node:pe=1 -np 30"
mc1.mc1step(C_no)
# data = lib.chkfile.load(mc1.chkfile, 'mcscf')
# mc1.mo_coeff = data["mo_coeff"]

# # Run final tighter HCI calculation (without orbital optimization) and generate
# # dice wavefunction.
mc1.fcisolver.sweep_epsilon = [1e-3, 1e-4, 5e-5]
mc1.fcisolver.sweep_iter = [0, 3, 5]
shci.dryrun(mc1)
shci.writeSHCIConfFile(mc1.fcisolver, mc1.nelecas, False)
with open(mc1.fcisolver.configFile, 'a') as f:
    f.write('noio\n')
    f.write('writebestdeterminants 1000000\n\n')
shci.executeSHCI(mc1.fcisolver)

# Generate QMC hamiltonian in full orbital space (in basis of SHCISCF mo
# coefficients.)
h1e, chol, enuc = generate_integrals(
        mol,
        mf.get_hcore(),
        mc1.mo_coeff,
        chol_cut=1e-5,
        verbose=True)

from ipie.utils.io import write_hamiltonian
h1eff, chol_act, e0_eff = freeze_core(h1e, chol, enuc, mc1.mo_coeff, ncore,
                                      verbose=True)
# after core averaging
write_hamiltonian(h1eff[0], chol_act, e0_eff)
