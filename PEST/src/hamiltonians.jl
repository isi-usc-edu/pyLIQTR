# Copyright (c) 2023 Massachusetts Institute of Technology 
# SPDX-License-Identifier: MIT


################################################################################
###                                                                          ### 
###      HAMILTONIANS  -  Methods for electronic structure Hamiltonians      ###         
###                                                                          ### 
################################################################################





###
###  STRUCT FermionicHamiltonian
### ============================================================================
###   
###   Data container for a fermionic Hamiltonian.
###

mutable struct FermionicHamiltonian

    spinless      ::  Bool                  # Flag for spinless/spinful fermions
                                            # (unused)
    periodic      ::  Bool                  # Flag for periodic system (unused)
  
    coeffs_fermi  ::  Bool                  # Flag for fermionic operator coeffs
    coeffs_pauli  ::  Bool                  # Flag for Pauli string coeffs

    n_orb         ::  Int64                 # Number of orbitals
    n_qb          ::  Int64                 # Number of required qubits

    ## Containers for coefficients of fermionic operator terms
    ##
    T         ::  Array{Float64,1}      # Coefficients of kinetic terms
    V         ::  Array{Float64,1}      # Coefficients of e-e interaction terms
#   U         ::  Array{Float64,1}      # Coefficients of e-ion interaction terms
    U         ::  Any

    ## Containers for coefficients of Pauli string terms
    ##
    c_i       ::  Float64               # Coefficients of identity terms
    c_z       ::  Array{Float64,1}      # Coefficeints of single Z terms
    c_zz      ::  Array{Float64,1}      # Coefficients of Z \otimes Z terms
    c_xzx     ::  Array{Float64,1}      # Coefficients of XZX / YZY terms

    # T_dict    ::  Dict
    # V_dict    ::  Dict
    # U_dict    ::  Array{Float64,1}

    # c_i_dict    ::  Float64           
    # c_z_dict    ::  Array{Float64,1}      
    # c_zz_dict   ::  Dict      
    # c_xzx_dict  ::  Dict     

    function FermionicHamiltonian(n_orb;spinless=true,periodic=true)

        ## Set number of orbitals required after JW transform
        ##
        if (spinless == false)
            n_qb = 2*n_orb
        else
            n_qb = n_orb
        end

        coeffs_fermi  =  true
        coeffs_pauli  =  false

        new(spinless,periodic,
            coeffs_fermi,coeffs_pauli,
            n_orb,n_qb)
    end
end





###
###  FUNCTION  ham, grid = calc_hamiltonian
### ======================================================================
###   
###  Wrapper to calculate an eletronic structure Hamltonian via PEST.  
### 
###
###     OPTIONS
###
###         filename  -  path to coordinate file 
###         pad       -  padding for unit cell (in angstrom)
###         E_cut     -  PW kinetic energy cutoff (in Rydberg). Converted 
###                      to grid spacing a0 if method="dpw".
###         a0        -  DPW grid spacing (in Bohr)
###         scale     -  Scale factor for DPW grid. 
### 
###                             scale = 1.0     Space grid points at equivalent
###                                             PW wavelength
###                             scale = 0.5     Space DPW grid points at half
###                                             of equivalent PW wavelength
###
###         jellium_terms   -  Calculate kinetic and electron-electron
###                            interaction terms (default = true)
###         ion_terms       -  Calculate electron-ion interaction terms
###                            (default = true)
###         pauli_strings   -  Calculate Pauli string coefficients via
###                            transform <pauli_method> (default = true)
###         pauli_method    -  Method to map fermions to Pauli strings 
###                            (default="jw"; only Jordan-Wigner supported now)
###         periodic        -  Assume periodic system (default = true; only true supported)
###                            Need to add support for cutoffs to define isolated molecule
###         method          -  Method for calculating Hamiltonian.  Currently only 
###                            dual plane-waves (DPW; method="dpw") is supported.  Need to
###                            add support for plane waves in the future.
###         eps_z           -  Numerical threshold below which a given term is set to zero.
###         lengths         -  Optional length of cell vector along each cartesian coordinate.
###                            Overrides default from cartesian coordinates.
###
###     RETURNS
###
###         ham             -  FermioncHamiltonian container for calculated Hamiltonian
###         grid            -  Real space grid / k-point mesh used for calculating 
###                            the Hamiltonian
###

function calc_hamiltonian(  filename,; 
                            pad=[0.0,0.0,0.0],
                            E_cut = nothing,
                            a0 = nothing,
                            scale=1.0,
                            jellium_terms=true,
                            ion_terms=true,
                            pauli_strings=true,
                            pauli_method="jw",
                            periodic=true,
                            method="dpw",
                            eps_z=1e-10,
                            lengths=nothing,
                            npts=nothing     )

    ##  Get geometry from atomic model, specify real/reciprocal-space grids
    ## 

    supplied_vectors = false

    println("###  Requested a ",uppercase(method)," Hamiltonian.\n")
    if (a0 != nothing)
        a0=a0
        println("\tGrid Spacing   =  ",round(a0,digits=6)," Bohr   (scaling = ",scale,")")
    elseif (E_cut != nothing)
        println("\tEnergy Cutoff  =  ",E_cut," Ry ")
        a0 = pw_to_dpw_cutoff(E_cut,dpw_units="bohr",scale=scale)
        println("\tGrid Spacing   =  ",round(a0,digits=6)," Bohr   (scaling = ",scale,")")
    end

    atoms, vectors    =  get_geometry(filename,pad=pad)

    if (lengths != nothing)
        vectors = [ [ lengths[1], 0.0, 0.0 ], [ 0.0, lengths[2], 0.0 ], [ 0.0, 0.0, lengths[3] ] ] * angstrom_to_bohr
    end

    if (npts == nothing)
        npts            =  get_npts(vectors,a0) 
    end
    
    println("\tGrid           =  ",npts[1]," x ",npts[2]," x ",npts[3],
                    "  ( ",prod(npts)," basis functions ).\n")

    print()
    println("###  Loaded geometry for ",atoms.n_atom," atoms.\n")
    print()
    println("\tLattice Dimensons  =  ",round(vectors[1][1],digits=3)," x ",
                                     round(vectors[2][2],digits=3)," x ",
                                     round(vectors[3][3],digits=3)," (Bohr)")

    spacing = calc_grid_spacing(vectors,npts)

    println("\tActual Spacing     =  ",round(spacing[1],digits=3)," x ",
                                       round(spacing[2],digits=3)," x ",
                                       round(spacing[3],digits=3)," (Bohr)\n")                          

    println("###  Building grid and mesh.\n")
    grid              =  get_grid(vectors;E_cut=E_cut,scale=scale)



    if (lowercase(method) == "dpw")
        ham  =  calc_hamiltonian_dpw(grid;atoms=atoms,
                                     jellium_terms=jellium_terms,
                                     ion_terms=ion_terms,
                                     pauli_strings=pauli_strings,
                                     eps_z=eps_z)

    elseif (lowercase(method) == "pw")
        ham  =  calc_hamiltonian_pw(grid;atoms=atoms,
                                     jellium_terms=jellium_terms,
                                     ion_terms=ion_terms,
                                     pauli_strings=pauli_strings,
                                     eps_z=eps_z,
                                     E_cut = E_cut)
    end


    # ##  Prepare Hamiltonian container for output
    # ##   
    # ham               =  FermionicHamiltonian(grid.N)
    # ham.coeffs_fermi  =  true


    # println("###  Calculating the Hamiltonian.\n")
    # ##  Calculate kinetic and electron-eletron interaction terms
    # ## 
    # if jellium_terms 
    #     println("\tCalculating Kinetic & E-E Interaction Terms:")
    #     flush(stdout)
    #     T,V    =  calc_electron_terms_dpw(grid,eps_z=eps_z);
    #     ham.T  =  T
    #     ham.V  =  V
    # end


    # ##  Calculate electron-ion interation terms
    # ## 
    # if ion_terms
    #     println("\tCalculating E-Ion Terms:")
    #     flush(stdout)
    #     U    =  calc_electron_ion_terms_dpw(grid,atoms,eps_z=eps_z);
    #     ham.U  =  U
    # end

    # ##  Also calculate Pauli-string coefficients, if requested.
    # ## 
    # if pauli_strings
    #     if !jellium_terms
    #         T = zeros(grid.N)
    #         V = zeros(grid.N)
    #     end
    #     if !ion_terms
    #         U = zeros(grid.N)
    #     end
    #     if (lowercase(pauli_method) == "jw")
    #         c_i, c_z, c_zz, c_xzx = pauli_string_coeffs_jw_dpw(grid,T,V,U);
    #     end
    #     ham.c_i    =  c_i
    #     ham.c_z    =  c_z
    #     ham.c_zz   =  c_zz
    #     ham.c_xzx  =  c_xzx  
    #     ham.coeffs_pauli = true
    # end

    return(ham,grid)
end





function calc_hamiltonian_dpw(grid;atoms=nothing,jellium_terms=true,ion_terms=true,pauli_strings=true,eps_z=1e-10)


    ##  Prepare Hamiltonian container for output
    ##   
    ham               =  FermionicHamiltonian(grid.N)
    ham.coeffs_fermi  =  true


    println("###  Calculating the Hamiltonian.\n")
    ##  Calculate kinetic and electron-eletron interaction terms
    ## 
    if jellium_terms 
        println("\tCalculating Kinetic & E-E Interaction Terms:")
        flush(stdout)
        T,V    =  calc_electron_terms_dpw(grid,eps_z=eps_z);
        ham.T  =  T
        ham.V  =  V
    end


    ##  Calculate electron-ion interation terms
    ## 
    ion_flag = ion_terms && (atoms != nothing)

    if (ion_flag)
        println("\tCalculating E-Ion Terms:")
        flush(stdout)
        U    =  calc_electron_ion_terms_dpw(grid,atoms,eps_z=eps_z);
        ham.U  =  U
    end

    ##  Also calculate Pauli-string coefficients, if requested.
    ## 

    if pauli_strings
        pauli_mapper(ham,grid,pauli_string_coeffs_jw_dpw,
                    jellium_terms=jellium_terms,
                    ion_terms=ion_terms)
    end

    return(ham)
end








function calc_hamiltonian_pw(grid;atoms=nothing,jellium_terms=true,ion_terms=true,pauli_strings=true,eps_z=1e-10,E_cut=nothing)


    ##  Prepare Hamiltonian container for output
    ##   
    ham               =  FermionicHamiltonian(grid.N)
    ham.coeffs_fermi  =  true


    println("###  Calculating the Hamiltonian.\n")
    ##  Calculate kinetic and electron-eletron interaction terms
    ## 
    if jellium_terms 
        println("\tCalculating Kinetic & E-E Interaction Terms:")
        flush(stdout)
        T,V    =  calc_electron_terms_pw(grid,eps_z=eps_z,E_cut=E_cut);
        ham.T  =  T
        ham.V  =  V
    end


    ##  Calculate electron-ion interation terms
    ## 
    ion_flag = ion_terms && (atoms != nothing)

    if (ion_flag)
        println("\tCalculating E-Ion Terms:")
        flush(stdout)
        U    =  calc_electron_ion_terms_pw(grid,atoms,eps_z=eps_z);
        ham.U  =  U
    end

    ##  Also calculate Pauli-string coefficients, if requested.
    ## 

    if pauli_strings
        pauli_mapper(ham,grid,pauli_string_coeffs_jw_pw,
                    jellium_terms=jellium_terms,
                    ion_terms=ion_terms)
    end

    return(ham)
end








function pauli_mapper(ham::FermionicHamiltonian,grid::GridData,map_function::Function,;jellium_terms=true,ion_terms=true)

        if !jellium_terms
            T = zeros(grid.N)
            V = zeros(grid.N)
        else
            T = ham.T
            V = ham.V
        end

        if !ion_terms
            U = zeros(grid.N)
        else
            U = ham.U
        end

        c_i, c_z, c_zz, c_xzx = map_function(grid,T,V,U);

        ham.c_i    =  c_i
        ham.c_z    =  c_z
        ham.c_zz   =  c_zz
        ham.c_xzx  =  c_xzx  
        ham.coeffs_pauli = true

end



###
###  FUNCTION  T_pq,V_pq = calc_electron_terms_dpw(grid)
###  FUNCTION  T_pq,V_pq = calc_electron_terms_dpw(grid,R_pq_table)
### ======================================================================
###   
###   Build terms for the "Jellium" part of the DPW electronic structure 
###   Hamiltonian.  Arguments are a Hamiltonian container <ham>, a 
###   grid defining the orbtials / basis functions <grid> and a distance
###   lookup table <R_pq_table> calculated in advance.  
###

function calc_electron_terms_dpw(grid::GridData,;eps_z=1e-10)
    
    T_pq,V_pq = calc_electron_terms_dpw(grid,grid.dtable_rs,eps_z=eps_z)
    return(T_pq,V_pq)
end


function calc_electron_terms_dpw(grid,R_pq;spinless=true,eps_z=1e-10)

    atol_k2  =  1e-16                # Tolerance parameter to avoid
                                     # singular behavior

    C_T      =  1.0/(2.0*grid.N)     # Kinetic term prefactor
    C_V      =  2*pi/(grid.vol)      # EE term prefactor

    if (spinless == false)
        C_T      =  0.5*CT
    end

    N_term   =  length(R_pq)

    T_pq     =  zeros(N_term)        # Kinetic terms
    V_pq     =  zeros(N_term)        # EE interaction terms


    ##  Loop over distinct lattice vectors and k-vectors to calculate
    ##  kinetic and electron-electron interaction potential coeffs
    ##
    prog = Progress(grid.N);
    update!(prog,0)
    tatm = Threads.Atomic{Int}(0)

    lock = Threads.SpinLock()

    # 

    Threads.@threads for n_d=1:N_term

        T    =  0.0         
        V    =  0.0

        for n_k=1:grid.N 

            ## Handle singular terms (which formally cancel)
            ##
            if (isapprox(grid.k2[n_k],0.0;atol=atol_k2) == false)

                cos_factor   =  cos(grid.k[n_k] ⋅ R_pq[n_d])
                    
                T     +=  (cos_factor * grid.k2[n_k])
                V     +=  (cos_factor / grid.k2[n_k])

            end
        end

        T_pq[n_d]  =  test_eps_z(C_T*T,eps_z=eps_z)
        V_pq[n_d]  =  test_eps_z(C_V*V,eps_z=eps_z)

        Threads.atomic_add!(tatm, 1)
        Threads.lock(lock)
        update!(prog, tatm[])
        Threads.unlock(lock)  

    end
    return(T_pq,V_pq)
end





function test_eps_z(val,;eps_z=1e-10)
    if (abs(val) < eps_z)
        return(0.0)
    else
        return(val)
    end
end


###
###  FUNCTION   U_p = calc_electron_ion_terms_dpw(grid,atoms)
### ======================================================================
###   
###   Build electron-ion interaction terms for the DPW Hamiltonian. Arguments 
###   are grid defining the basis functions / the distance lookup table and
###   a set of atomic coordinates via an Atoms container in <atoms>.
###
###   This does a naive summation, so it is very slow (should replace with
###   more intelligend Coulomb / Ewald sum methods)
###

function calc_electron_ion_terms_dpw(grid,atoms;eps_z=1e-10)
    
    atol_k2  =  1e-16                # Tolerance parameter to avoid
                                     # singular behavior

    U_p      =  zeros(grid.N)
    C_U      =  -4.0*pi/grid.vol


    prog = Progress(grid.N);
    update!(prog,0)
    tatm = Threads.Atomic{Int}(0)

    lock = Threads.SpinLock()

    Threads.@threads for n_p=1:grid.N

        r_p    =  grid.x[n_p]
        U      =  0.0;  

        for n_k=1:grid.N
            if (isapprox(grid.k2[n_k],0.0;atol=atol_k2) == false)
                for n_i=1:atoms.n_atom        
                    U += ( atoms.atnum[n_i] * cos(grid.k[n_k] ⋅ 
                                (atoms.coords[n_i,:] - r_p)) ) / grid.k2[n_k]
                end
            end
        end

        U_p[n_p] = test_eps_z(C_U*U,eps_z=eps_z)

        Threads.atomic_add!(tatm, 1)
        Threads.lock(lock)
        update!(prog, tatm[])
        Threads.unlock(lock)  
    end

    return(U_p)
end





###
###  FUNCTION  T_p,V_nu = calc_electron_terms_pw(grid)
###  FUNCTION  T_p,V_nu = calc_electron_terms_pw(grid,R_pq_table)
### ======================================================================
###   
###   Build coefficients for the "Jellium" part of the PW electronic structure 
###   Hamiltonian.  Arguments are a grid defining the k-point mesh and 
###   (optional) the energy cutoff <E_cut> if this is to be strictly 
###   enforced (in Rydberg)
###

function calc_electron_terms_pw( grid::GridData,;
                                 spinless::Bool=true,
                                 eps_z::Float64=1e-10,
                                 E_cut=nothing )


    if (E_cut == nothing)
        E_cut = 1e16
    else
        E_cut = 0.5*E_cut
    end

    C_V      =  2*pi/(grid.vol)      # EE term prefactor

    N_term   =  grid.N

    T_p      =  zeros(N_term)        # Kinetic terms
    V_nu     =  zeros(N_term)        # EE interaction terms

    pcut = grid.vol^(1/3)            # GENERALIZE
    pscale   =  1.0


    ## Calculate k_p^2 / 2 for all points in reciprocal lattice
    ##

    prog = Progress(N_term);
    update!(prog,0)
    tatm = Threads.Atomic{Int}(0)

    lock = Threads.SpinLock()

    Threads.@threads for n_d=1:N_term

        T  =  0.5*grid.k2[n_d]

        if (T <= E_cut)
            T_p[n_d] = T
        end

        ## Aperiodic grid scaling assuming cubic, isotropic grid (generalize)
        ##
        if !grid.periodic             
            pscale = 1.0 - cos(pcut*sqrt(grid.k2[n_d]))
        else
            pscale = 1.0
        end

        V_nu[n_d] = pscale*C_V/grid.k2[n_d] 

        Threads.atomic_add!(tatm, 1)
        Threads.lock(lock)
        update!(prog, tatm[])
        Threads.unlock(lock)  

    end

    return(T_p,V_nu)

end





function calc_electron_ion_terms_pw( grid::GridData,
                                     atoms::Atoms,;
                                     eps_z=1e-10     )
    
    atol_k2  =  1e-16                # Tolerance parameter to avoid
                                     # singular behavior


    N_term   =  length(grid.dtable)

    U_pq     =  zeros(Complex{Float64},N_term)
    C_U      =  -4.0*pi/grid.vol


    prog = Progress(N_term);
    update!(prog,0)
    tatm = Threads.Atomic{Int}(0)
    lock = Threads.SpinLock()

    dpq_to_kpq  =  2*pi*[1.0/grid.vecs[1][1], 1.0/grid.vecs[2][2], 1.0/grid.vecs[3][3]]

    Threads.@threads for n_d=1:N_term

        ##
        ## Is periodicity being handled corectly here? (should be ok since distance...)
        ##

        U      =  0.0         

        k_pq   =  grid.dtable_rs[n_d].*dpq_to_kpq
        k2_pq  =  k_pq ⋅ k_pq 

        for n_i=1:atoms.n_atom 

            if (isapprox(k2_pq,0.0;atol=atol_k2) == false)

                U += atoms.atnum[n_i] * exp(im * k_pq ⋅ atoms.coords[n_i,:]) / k2_pq
            
            end
        end

        U_pq[n_d]  =  test_eps_z(C_U*U,eps_z=eps_z)

        Threads.atomic_add!(tatm, 1)
        Threads.lock(lock)
        update!(prog, tatm[])
        Threads.unlock(lock)  

    end

    return(U_pq)

end






###
###  FUNCTION   c_i, c_z, c_zz, c_xzx = fermionic_to_qubit_coefficients
### ======================================================================
###   
###   Convert coefficients for a dual plane-wave (DPW) Hamiltonian written 
###   in terms of fermionic operators into one written in terms of quibits.
###   The Hamiltonian <ham> and grid <grid> match that given to a coefficent
###   calculation, while the kinetic <T_pq>, electron-electron potential
###   <V_pq> and electron-ion potential <U_p> correspond to its output.
###

function pauli_string_coeffs_jw_dpw(grid,T_pq,V_pq,U_p)

    k2_sum = 0.0

    for val in grid.k2
        k2_sum += val
    end

    c_i    =  k2_sum/2.0 - pi*grid.N/(grid.vol * k2_sum)
    c_z    =  pi/(grid.vol*k2_sum) .- k2_sum/(4*grid.N) .- U_p/ 2.0
    c_zz   =  V_pq / 4.0 
    c_xzx  =  T_pq / 2.0

    return( c_i, c_z, c_zz, c_xzx )

end





###
###  FUNCTION   E_cut = pw_to_dpw_cutoff()
### ======================================================================
###   
###   Calculate the DPW lattice spacing cutoff corresponding to the 
###   PW energy cutoff of <E_cut_inp> specified with units <pw_units>. 
###   The DPW spacing is returned with units <dpw_units>.  The factor 
###   <scale> is used to tune the spacing.  For instance, setting
###   scale=0.5 will place an additional basis function between each
###   point in the direct mapping (i.e., it haves the real-space cutoff)
###

function pw_to_dpw_cutoff(E_cut_inp,;scale=1.0,dpw_units="Bohr",pw_units="Ry")


    if ( (lowercase(pw_units) == "ry") || (lowercase(pw_units) == "rydberg") )
        ## Convert Rydberg to Hartree
        ##
        E_cut  =  E_cut_inp/2
    elseif ( (lowercase(pw_units) == "ha") || (lowercase(pw_units) == "hartree") )
        E_cut  =  E_cut_inp
    elseif ( (lowercase(pw_units) == "ev") )
        E_cut  =  E_cut_inp*ev_to_hartree
    end


    ## Set scale to 0.5 to sample node between points, otherwise set to 1
    ## to sample exactly at spacing.
    ##
    a0_bohr      =  sqrt(2*pi^2/E_cut)*scale

    if (lowercase(dpw_units) == "bohr")
        return(a0_bohr)
    elseif (lowercase(dpw_units) == "angstrom")
        return(a0_bohr*bohr_to_angstrom)
    end

end





###
###  FUNCTION   E_cut = dpw_to_pw_cutoff()
### ======================================================================
###   
###   Calculate the PW energy  cutoff corresponding to a DPW grid with 
###   lattice spacing of <a0_inp> specified with units <dpw_units>. 
###   The DPW spacing is returned with units <pw_units>.  The factor 
###   <scale> is used to tune the spacing (see pw_to_dpw_cutoff)
###

function dpw_to_pw_cutoff(a0_inp,;scale=1.0,dpw_units="Bohr",pw_units="Ry")


    if (lowercase(dpw_units) == "bohr")
        a0_bohr = a0_inp
    elseif (lowercase(dpw_units) == "angstrom")
        a0_bohr = a0_inp*angstrom_to_bohr
    end

    E_cut = (2*pi^2)/(a0_bohr/scale)^2


    if ( (lowercase(pw_units) == "ry") || (lowercase(pw_units) == "rydberg") )
        E_cut_out  =  E_cut*2
    elseif ( (lowercase(pw_units) == "ha") || (lowercase(pw_units) == "hartree") )
        E_cut_out  =  E_cut
    elseif ( (lowercase(pw_units) == "ev") )
        E_cut_out  =  E_cut/ev_to_hartree
    end

    return(E_cut_out)

end





###
###  FUNCTION   val = pair_coeff_lookup(p,q,grid::GridData,T)
### ======================================================================
###   
###   Look up pair coefficients from array <T> for scalar (qubit) indicies 
###   <p>,<q> according to the grid in <grid>
###

function pair_coeff_lookup(p::Int64,q::Int64,grid::GridData,T)

    if  (grid.dtable_dict == nothing)
        make_dtable_dict(grid)
    end

    if (p == q)
        val = 0.0
    else
        d_pq    =  grid.id_to_crd[p] - grid.id_to_crd[q]
        idx     =  grid.dtable_dict[d_pq]
        val     =  T[idx]
    end
    return(val)
end










# function all_terms_dict(grid,dtable,T)

#     T_pq  = Dict()

#     dtable_dict_p      =  Dict( dtable[i] => i for i = 1:length(dtable))
#     dtable_dict_m      =  Dict(-dtable[i] => i for i = 1:length(dtable))

#     dtable_dict        =  merge(dtable_dict_p,dtable_dict_m)

#     for p=1:grid.N
#         T_pq[p,p]      =  0.0
#     end

#     for p=1:grid.N
#         for q=(p+1):grid.N
#             d_pq            =  grid.id_to_crd[p] - grid.id_to_crd[q]
#             T_pq[[p,q]]     =  T[dtable_dict[d_pq]]
#         end
#     end

#     return(T_pq)
# end


# function all_terms_dict(grid::GridData,T::Array{Float64,1})
#     T_pq = all_terms_dict(grid,grid.dtable,T)
#     return(T_pq)
# end



# function make_term_dicts(ham,grid,;pauli=true,fermi=true)

#     if (fermi && ham.coeffs_fermi)
#         ham.T_dict = all_terms_dict(grid,grid.dtable,ham.T);
#         ham.V_dict = all_terms_dict(grid,ham.V);
#         ham.U_dict = ham.U
#     end

#     if (pauli && ham.coeffs_pauli)
#         ham.c_i_dict = ham.c_i
#         ham.c_z_dict = ham.c_z
#         ham.c_zz_dict  = all_terms_dict(grid,ham.c_zz);
#         ham.c_xzx_dict = all_terms_dict(grid,ham.c_xzx);
#     end

# end








