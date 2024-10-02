# Copyright (c) 2023 Massachusetts Institute of Technology 
# SPDX-License-Identifier: MIT


################################################################################
###                                                                          ### 
###      PEST COMMANDER  -  Command-line ethods for electronic structure     ###         
###                                                                          ### 
################################################################################


push!(LOAD_PATH,pwd())

using PEST


function main()


    E_cut  =  30    
    scale  =  1.0
    pad    =  [0.0,0.0,0.0]

    println() 


    println("PEST COMMANDER    ( Threads = ",Threads.nthreads()," )\n")
    

    if (length(ARGS) == 0) 
        println("PEST:  No input geometry specified!  Exiting.\n")
        exit()
    end


    (length(ARGS) >= 1) &&  (filename = ARGS[1])
    (length(ARGS) >= 2) &&  (E_cut    = parse(Float64,ARGS[2]))
    (length(ARGS) >= 3) &&  (scale    = parse(Float64,ARGS[3]))

    (length(ARGS) >= 4) &&  (pad[1]   = parse(Float64,ARGS[4]))
    (length(ARGS) >= 4) &&  (pad[1]   = parse(Float64,ARGS[5]))
    (length(ARGS) >= 4) &&  (pad[1]   = parse(Float64,ARGS[6]))


    ham,grid = calc_hamiltonian( filename,
                pad=pad,E_cut = E_cut,scale=scale   );

    write_hamiltonian_hdf5("ham.hdf5",ham)
    write_grid_hdf5("grid.hdf5",grid)

    println("\n###  Writing hamiltonian and grid in HDF5 format.\n")

    print("DONE!!\n\n")

end

main()