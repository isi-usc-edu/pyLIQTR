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
### ============================================================================
###   FUNCTION  write_hamiltonian
### ============================================================================
###
### DESCRIPTION
###
###     A wrapper for Hamiltonian IO operations. Writes Hamiltonian <ham> with  
###     filename <filename> in in format <format>.  Overwrite is automatic.
###
### ARGUMENTS
###
###     <filename>      -  Target filename (full path allowed)
###     <ham>           -  FermiHamiltonian structure
###     <format>        -  Output format:
###
###                         "hdf5"   -  Binary HDF5 file (.hdf5 extension)
###                                        
###                         "ascii"  -  A set of plain-text files, one for each 
###                                     field in the Hamiltonian (.dat extension)
###
### RETURNS
###
###       NONE
###


###
###  write_hamiltonian
### ======================================================================
###   
###   Wrapper for Hamiltonian IO.  Writes Hamiltonian <ham> with filename
###   <filename> in in format <format>.  Supported formats are 
###   ["ascii", "hdf5"], with "hdf5" writing a single binary file and 
###   "ascii" writing a different plain-text file for each set of terms.
###

function write_hamiltonian(filename,ham,;format="hdf5")

    if (lowercase(format) == "hdf5")
        write_hamiltonian_hdf5(filename,ham)
    elseif (lowercase(format) == "ascii")
        write_hamiltonian_ascii(filename,ham)
    end

end





###
###  write_hamiltonian_hdf5
### ======================================================================
###   
###   Writes Hamiltonian <ham> to  <filename> as a binary HDF5 file.  
###   The HDF5 keys and outputs should be apparent from the code below.
###
###     OVERWRITE PROTECTION NOT IMPLEMENTED YET
###

function write_hamiltonian_hdf5(filename,ham,;overwrite=true)

    ## Should eventually check for file to ensure no 
    ## overwite unless requested.
    ##
    fp = h5open(filename,"w")

    fp["n_qb"]          =  ham.n_qb
    fp["n_orb"]         =  ham.n_orb
    fp["spinless"]      =  ham.spinless
    fp["coeffs_fermi"]  =  ham.coeffs_fermi
    fp["coeffs_pauli"]  =  ham.coeffs_pauli

    if (ham.coeffs_fermi)
        fp["T"]         =  ham.T
        fp["U"]         =  ham.U
        fp["V"]         =  ham.V
    end

    if (ham.coeffs_pauli)
        fp["c_i"]       =  ham.c_i
        fp["c_z"]       =  ham.c_z
        fp["c_zz"]      =  ham.c_zz
        fp["c_xzx"]     =  ham.c_xzx
    end

    close(fp)

end




###
###  read_hamiltonian_hdf5
### ======================================================================
###   
###   Read a saved HDF5 format Hamiltonian from file <filename>. Returns a
###   Hamiltonian container.
###

function read_hamiltonian_hdf5(filename)

    fp = h5open(filename,"r")

    n_qb      =  read(fp["n_qb"])
    n_orb     =  read(fp["n_orb"])
    spinless  =  read(fp["spinless"])

    ham = FermionicHamiltonian(n_orb,spinless=spinless)

    ham.n_qb  =  n_qb

    ham.coeffs_fermi  =  read(fp["coeffs_fermi"])
    ham.coeffs_pauli  =  read(fp["coeffs_pauli"])

    if (ham.coeffs_fermi)
        ham.T         =  read(fp["T"])
        ham.U         =  read(fp["U"])
        ham.V         =  read(fp["V"])

    end

    if (ham.coeffs_pauli)
        ham.c_i       =  read(fp["c_i"])   
        ham.c_z       =  read(fp["c_z"])   
        ham.c_zz      =  read(fp["c_zz"])   
        ham.c_xzx     =  read(fp["c_xzx"])  
    end

    close(fp)

    return(ham)

end















function write_grid_hdf5(filename,grid,;overwrite=true,verbose=false)

    ## Should eventually check for file to ensure no 
    ## overwite unless requested.
    ##
    fp = h5open(filename,"w")


    fp["n_points"]      =  grid.n_points
    fp["vecs"]          =  hcat(grid.vecs...)
    fp["dtable"]        =  hcat(grid.dtable...)

    if (verbose == true)
        fp["dim"]           =  dim
        fp["N"]             =  N 

        fp["spacing"]       =  grid.spacing

        fp["kvecs"]         =  hcat(grid.kvecs...)
        fp["vol"]           =  grid.vol

        fp["id_to_crd"]     =  hcat(grid.id_to_crd...)

        fp["x"]             =  hcat(grid.x...)
        fp["k"]             =  hcat(grid.k...)
        fp["k2"]            =  grid.k2

        fp["dtable_rs"]     =  hcat(grid.dtable_rs...)
    end

    close(fp)

end






function read_grid_hdf5(filename)

    fp = h5open(filename,"r")

    n_points  =  read(fp["n_points"])
    vectors   =  read(fp["vecs"])

    vecs      =  []

    for n=1:3
        push!(vecs,vectors[:,n])
    end

    grid = GridData(n_points,vecs)

    grid.dtable     =  read(fp["dtable"])
    grid.dtable_rs  =  calc_distances_physical(grid.dtable,grid.spacing);

    close(fp)

    return(grid)

end









###
###  write_dtable_hdf5
### ======================================================================
###   
###   Write the distance table <R_pq_table> to a file <filename.  An entry
###   <npts> is added for the number of basis functions along each axis
###   to check for compatibility on read.
###
###     HDF5 fields
###
###         npts    -   Array containing number of basis points along 
###                     each axis [ Array{Int64,1} ]
###         dkeys   -   Integer keys indexing each basis point pair
###                     [ Array{Int64,1} ]
###         dvals   -   Two-dimensional float array. Each column contains
###                     a displacement vector tagged to the corresponding
###                     dkeys value (row index dvals -> index in kdkeys)
###

function write_dtable_hdf5(filename,R_pq_table,npts)

    ## Should eventually check for file to ensure no 
    ## overwite unless requested.
    ##
    fp = h5open(filename,"w")

    fp["npts"]          =  npts

    dkeys = collect(keys(R_pq_table))
    dvals = collect(values(R_pq_table))

    dkeys = convert(Array{Int64,1}, dkeys)
    dvals = convert(Array{Array{Float64,1}}, dvals)
    dvals = hcat(dvals...)

    fp["dkeys"]         =  dkeys
    fp["dvals"]         =  dvals

    close(fp)

end




###
###  write_dtable_hdf5
### ======================================================================
###   
###   Read distance table (dict) from hdf5 file <filename>.
###

function read_dtable_hdf5(filename)

    ## Should eventually check for file to ensure no 
    ## overwite unless requested.
    ##
    fp = h5open(filename,"r")

    npts        =  read(fp["npts"])
    dkeys       =  read(fp["dkeys"])
    dvals_mat   =  read(fp["dvals"])

    dvals = []

    for n in 1:(size(dvals_mat)[2])
        push!(dvals,dvals_mat[:,n])
    end
    
    R_pq_table  =  Dict()

    for pair in zip(dkeys,dvals)  
        R_pq_table[pair[1]] = pair[2]
    end

    # for n in size(dkeys)[1]
    #     R_pq_table[dkeys[n]] = dvals[n]
    # end

    close(fp)

    return(R_pq_table,npts)
    
end












###
###  write_hamiltonian_ascii
### ======================================================================
###   
###   Writes Hamiltonian <ham> to  <filename> as set of plain-text files.
###   The filenames and contents should be apparent from the code below.
###
###     OVERWRITE PROTECTION NOT IMPLEMENTED YET
###
    
function write_hamiltonian_ascii(prefix,ham,;overwrite=true)

    write_array_data(prefix * "_n_orb.dat",ham.n_qb)
    write_array_data(prefix * "_n_qb.dat",ham.n_orb)

    if (ham.coeffs_fermi)
        write_array_data(prefix * "_T.dat",ham.T)
        write_array_data(prefix * "_U.dat",ham.U)
        write_array_data(prefix * "_V.dat",ham.V)
    end

    if (ham.coeffs_pauli)
        write_array_data(prefix * "_I.dat",ham.c_i)
        write_array_data(prefix * "_Z.dat",ham.c_z)
        write_array_data(prefix * "_ZZ.dat",ham.c_zz)
        write_array_data(prefix * "_XZX.dat",ham.c_xzx)
    end

end





###
###  write_array_data
### ======================================================================
###   
###   Write data array <dataset> to plain-text file <filename>.  Write mode
###   is controlled by <mode>, with overwrite as a default.
###

function write_array_data(filename::String,dataset::Any;mode::String = "w+")
    open(filename,mode) do io
        writedlm(io,dataset," ")
    end
end



