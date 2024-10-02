import pytest

from    pyLIQTR.ProblemInstances.getInstance         import   getInstance
from    pyLIQTR.clam.lattice_definitions             import   SquareLattice
from    pyLIQTR.BlockEncodings.getEncoding           import   getEncoding, VALID_ENCODINGS

from    pyLIQTR.qubitization.qsvt_dynamics           import   qsvt_dynamics, simulation_phases
from    pyLIQTR.qubitization.phase_estimation        import   QubitizedPhaseEstimation


@pytest.fixture()
def fermi_hubbard_dynamics():
    J      = -1.0;          N      =     3     
    U      =  4.0;          shape  =  (N,N)

    model  =  getInstance('FermiHubbard',shape=shape, J=J, U=U, cell=SquareLattice)

    times       =  1
    eps         =  1e-1
    phases      =  simulation_phases(times,eps=eps)

    gate_qsvt   =   qsvt_dynamics( encoding=getEncoding(VALID_ENCODINGS.FermiHubbardSquare),
                            instance=model,
                            phase_sets=phases )
    return gate_qsvt.circuit

@pytest.fixture
def heisenberg_phase_estimation():
    N=3
    N_prec = 4

    J_x = J_y = -0.5;                J_z = -1.0
    h_x = 1.0;      h_y = 0.0;       h_z = 0.5

    model  =  getInstance( "Heisenberg", 
                        shape=(N,N), 
                        J=(J_x,J_y,J_z), 
                        h=(h_x,h_y,h_z), 
                        cell=SquareLattice )

    gate_gsee = QubitizedPhaseEstimation( getEncoding(VALID_ENCODINGS.PauliLCU),
                                    instance=model,prec=N_prec )
    
    return gate_gsee.circuit

@pytest.fixture()
def electronic_structure_LinearT_encoding(script_loc):
    example_ham_filename = script_loc.joinpath('example.ham.hdf5')
    example_grid_filename = script_loc.joinpath('example.grid.hdf5')

    model  =  getInstance('ElectronicStructure',
                            filenameH=example_ham_filename,
                            filenameG=example_grid_filename)

    encoding = getEncoding(VALID_ENCODINGS.LinearT, 
                            instance=model, 
                            approx_error=0.001,
                            control_val=1)
    
    return encoding.circuit
    