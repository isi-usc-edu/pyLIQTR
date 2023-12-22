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