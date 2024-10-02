# Copyright (c) 2023 Massachusetts Institute of Technology 
# SPDX-License-Identifier: MIT


################################################################################
###                                                                          ### 
###              PEST  -  pyLIQTR's Electronic Structure Toolkit             ###         
###                                                                          ### 
################################################################################


###
###  MODULE   PEST
### ============================================================================
###   
###   Module containing the PEST toolkit.  
###

module PEST

    using  ArgParse
    using  Combinatorics
    using  DelimitedFiles
    using  HDF5
    using  LinearAlgebra
    using  ProgressMeter

    import Base.Threads.@spawn
    import Base.Threads.@threads

    include("atoms.jl")                 # Coordinate IO and manipulation
    include("grids.jl")                 # Real-space grids and k-point meshes
    include("hamiltonians.jl")          # Calculation drivers for Hamiltonians
    include("io_functions.jl")          # General (non-atom) IO routines
    include("unit_conversion.jl")       # Common unit conversion factors


    ## Geometry specification and manipulation
    ##
    export Atoms, get_geometry, load_coordinates, load_xyz

    ## Grid utilities and wrappers 
    ##
    export GridData
    export build_distance_table, calc_distances_physical
    export unique_distance_dict

    ## Hamiltonian routines, utilities, and calulation wrappers
    ##
    export FermionicHamiltonian
    export calc_hamiltonian
    export calc_electron_terms_dpw, calc_electron_ion_terms_dpw
    export pauli_string_coeffs_jw_dpw
    export pair_coeff_lookup

    ## Input specifiation / preparation tools
    ##
    export pw_to_dpw_cutoff, dpw_to_pw_cutoff
    export get_npts, calc_grid_spacing 
    export pad_vectors

    ## IO Methods
    ##
    export read_hamiltonian_hdf5, write_hamiltonian_hdf5
    export read_grid_hdf5, write_grid_hdf5
    export write_hamiltonian_ascii

    ## Unit conversions
    ##
    export bohr_to_angstrom, angstrom_to_bohr, ev_to_hartree

end 
