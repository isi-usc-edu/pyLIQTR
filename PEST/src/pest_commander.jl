"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2022 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""

################################################################################
###                                                                          ### 
###      PEST COMMANDER  -  Command-line ethods for electronic structure     ###         
###                                                                          ### 
################################################################################


PEST_DIR=ENV["PEST_DIR"]

push!(LOAD_PATH,PEST_DIR)

using PEST
using ArgParse

function parse_inputs()

    s = ArgParseSettings()

    @add_arg_table s begin
        "--coords"
            help = "Path to coordinate file in XYZ format (units in Angstroms).  Cell vectors calculated from minimal bounding box in cartesian frame, unless --lenghts are specified."
            arg_type = String
            default = ""
        "--e_cut"
            help = "Plane-wave cutoff energy E_cut [Rydberg]"
            arg_type = Float64
            default = 30.0
        "--a0"
            help = "Real-space grid spacing a0 [Bohr]"
            arg_type = Float64
        "--scale"
            help = "Rescaling factor when real-space lattice constant is derived from E_cut [ a0' = scale * a0 ]"
            arg_type = Float64
            default = 1.0
        "--pad"
            help = "Vacuum padding added to supercell lengths (if derived from input coordinates)  [Angstrom]"
            arg_type = Float64
            nargs = 3
        "--lengths"
            help = "Lengths defining supercell along each cartesian coordinate [Angstrom]"
            arg_type = Float64
            nargs = 3
        "--eps_z"
            help = "Specify cutoff for rounding small coefficients to zero (default = 1e-10)"
            arg_type = Float64
            default = 1e-10
        # "--dpw"
        #     help = "Flag to calculate a (real-space) dual plane-pave Hamiltonian (default = true)"
        #     action = :store_true
        # "--pw"
        #     help = "Flag to calculate a plane-wave Hamiltonian (default = false)"
        #     action = :store_true
        # "--aperiodic"
        #     help = "Specify an isolated molecule (aperiodic system; default = false)"
        #     action = :store_true
        # "--calc_pauli"
        #     help = "Also calculate coefficients for Pauli strings (default = false)"
        #     action = :store_true
        "--output_pfx"
            help = "Prefix for grid and hamiltonian output files."
            arg_type = String
            default  = nothing
    end

    return parse_args(s)

end


function main()



    parsed_args = parse_inputs()

    ## Debugging stuff
    ##
    # for (arg,val) in parsed_args
    #     println("  $arg  =>  $val")
    # end

    E_cut       =  parsed_args["e_cut"]
    scale       =  parsed_args["scale"]
    a0          =  parsed_args["a0"]

    lengths     =  parsed_args["lengths"]
    pad         =  parsed_args["pad"]

    eps_z       =  parsed_args["eps_z"]
 #   calc_pauli  =  parsed_args["calc_pauli"]

    filename    =  parsed_args["coords"]
    output_pfx  =  parsed_args["output_pfx"]


    if (output_pfx != nothing)
        output_pfx = output_pfx*"."    
    else
        output_pfx = ""
    end
        

    if (length(parsed_args["pad"]) != 3)
        pad = [0.0,0.0,0.0]
    else
        pad = parsed_args["pad"]
    end


    if (length(lengths) != 3)
        lengths = nothing
    end

    println() 
    println("PEST COMMANDER    ( Threads = ",Threads.nthreads()," )\n")
    

    ham,grid = calc_hamiltonian( filename, pad=pad, E_cut = E_cut, a0=a0, scale=scale, lengths=lengths, eps_z=eps_z   );

    println("\n###  Writing hamiltonian and grid in HDF5 format.\n")

    write_hamiltonian_hdf5(output_pfx*"ham.hdf5",ham)
    write_grid_hdf5(output_pfx*"grid.hdf5",grid)


    print("DONE!!\n\n")

end

main()