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
###     GRIDS  -  Methods for real-space grids and recipocal-space meshes    ###         
###                                                                          ### 
################################################################################




###
###  GridData(n_points,vecs;centered=true)
### ======================================================================
###   
###   Data container for a real-space or k-point mesh within the simulation 
###   supercell.  Arguments are an array <n_points> containing the number of points
###   along each basis vector, an array of basis vectors <vecs> and a flag <centered> 
###   that controls centering of the mesh about the orgin. We assume a fully periodic
###   mesh for now (flagged internally) though this should be changed to allow slab
###   geometries, rods, etc.
###


mutable struct GridData

    dim         ::  Int64               # Dimensionality of the mesh (spatially)
    n_points    ::  Any                 # Number of points along each axis of the
                                        #   mesh (i.e., along each basis vector)
    periodic    ::  Bool
    vecs        ::  Any                 # Real space vectors 
    kvecs       ::  Any                 # Receptirocal lattice vectors
    N           ::  Int64               # Total number of points in the mesh
    id_to_crd   ::  Any                 # Array to hold indices of mesh points
    crd_to_id   ::  Dict                # Scalar ids for each set of grid point indices
    vol         ::  Float64             # Volume factor
    x_dict      ::  Dict                # Real-space vector for each point in mesh
                                        #   (lookup Dict in terms of ids)
    x           ::  Any                 #
    k_dict      ::  Dict                # K-space vector for each point in mesh
                                        #   (lookup Dict in terms of ids)
    k           ::  Any                 #
    k2_dict     ::  Dict                # |k|^2 for each k-vector
                                        #   (lookup Dict in terms of ids)
    k2          ::  Any                 #
    crd_pdict   ::  Dict

    spacing     ::  Any

    dtable_dict ::  Any

    dtable      ::  Any
    dtable_rs   ::  Any

    function GridData(n_points,vecs;centered=true)

        dim          =  length(n_points)
        N            =  prod(n_points)

        periodic     =  true

        if (centered)
            shifts = div.(n_points,2)
        else
            shifts = zeros(dim)
        end

        id_to_crd    =  index_set(n_points.-1;min=0)  # index set for grid coordinates
        
        vecs_mat   =  reduce(vcat,transpose.(vecs))   # array of real-space basis vectors
        kvecs_mat  =  2*pi*inv(vecs_mat)'             # array of recpirocal-space basis vectors
        vol        =  abs(det(vecs_mat))              # volume of cell

        kvecs      =  [kvecs_mat[:,i] for i in 1:size(kvecs_mat,2)]

        ## Build real- and recepirocal-space vector dicts / tables
        ##
        crd_to_id   =  Dict(id_to_crd[i] => i for i = 1:N)
        crd_pdict   =  x_vec_dict_id(id_to_crd,vecs,shifts,n_points,dimensionless=true)

        x_dict      =  x_vec_dict_id(id_to_crd,vecs,shifts,n_points)
        x           =  [x_dict[i] for i=1:N]

        k_dict      =  Dict(i => k_vec(id_to_crd[i],shifts,kvecs,
                                    n_points,periodic) for i = 1:N)
        k           =  [k_dict[i] for i=1:N]

        k2_dict     =  Dict(i => norm(k[i])^2 for i = 1:N)
        k2          =  [k2_dict[i] for i=1:N]

        spacing     =  calc_grid_spacing(vecs,n_points)

        dtable_dict =  nothing

        new(dim,n_points,periodic,vecs,kvecs,N,id_to_crd,crd_to_id,vol,x_dict,x,k_dict,k,k2_dict,k2,crd_pdict,spacing,dtable_dict) 
    end
end





###
###  set = index_set(npts;min=1)
### ======================================================================
###   
###   Build index set for points on a grid.  The maximal number along a
###   dimension is sepcified by components of the vector <npts>.  The
###   minimal index value is specified by <min>.  The return is an 
###   array of tuples <set>.
###

function index_set(npts;min=1)
    set = [[x,y,z] for x in min:npts[1] for y in min:npts[2] for z in min:npts[3]]
    return(set)
end





###
###  atnum, coords = k_vec(idx,shifts,cell_kvecs,n_points,periodic)
### ======================================================================
###   
###   Load atomic coordinates from xyz format file <filename>.  Returns an 
###   arrays <atnum> containing the atomic number a given atom and and 
###   <coords> containing the cartesian coordinates.  No unit conversion 
###   is performed so the resulting units match the input file.
###

function k_vec(idx,shifts,cell_kvecs,n_points,periodic)
    dim = length(idx)
    kidx = idx - shifts
    if (periodic)
        kidx = ((kidx + shifts) .% n_points) - shifts
    end
    kvec = zeros(dim)
    for (i,val) in enumerate(kidx)
        kvec .+= val*cell_kvecs[i]
    end
    return(kvec)
end





###
###  theDict = x_vec_dict(idxs,vecs,shifts,n_points)
### ======================================================================
###   
###   Create a dictionary for coordinates on a Euclidean real-space grid.
###   Indicies are provided as <idxs>, with supercell vectors <vecs> and
###   displacements <shifts>.  The number of points along the cell vectors 
###   is given by the array <n_points>.
###

function x_vec_dict(idxs,vecs,shifts,n_points)
    theDict = Dict()
    for (n,idx) in enumerate(idxs)
        x = zeros(3)
        for (i,val) in enumerate(idx)
            x .+= (val .- shifts[i]) .* vecs[i] ./ n_points[i] 
        end
        theDict[idx] = x
    end
    return(theDict)
end





###
###  theDict = x_vec_dict(idxs,vecs,shifts,n_points)
### ======================================================================
###   
###   Create a dictionary for the raw coordinates of a real-space grid.
###   Indicies are provided as an array <id_to_crd> which associates a 
###   scalar index with each tuple specifying a grid point.
###   Make this optional: Supercell vectors <vecs> and
###   displacements <shifts>.  The number of points along the cell vectors 
###   is given by the array <n_points>.
###

function x_vec_dict_id(id_to_crd,vecs,shifts,n_points,;dimensionless = false)

    theDict = Dict()

    for (n,idx) in enumerate(id_to_crd)
        x = zeros(3)
        for (i,val) in enumerate(idx)
            if (dimensionless == false)
                x .+= (val .- shifts[i]) .* vecs[i] ./ n_points[i] 
            else
                x .+= (val .- shifts[i]) 
            end
        end
        theDict[n] = x
    end
    return(theDict)
end





###
###  vals = minmax(coords)
### ======================================================================
###   
###   Return an array containing the minimum and maximum values of the 
###   coordinate set in array <coords>.  
###

function minmax(coords)

    vals = zeros(3,2)

    for n=1:3
        vals[n,1] = minimum(coords[:,n])
        vals[n,2] = maximum(coords[:,n])
    end

    return(vals)
end





###
###  FUNCTION grid = get_grid.
### ======================================================================
###   
###   Generate a GridData object for a real-space grid / k-point mesh
###   associated with a spacing of <a0> or (equivalently) a plane-wave 
###   cutoff of <E_cut>.  The lattice vectors <vectors> must be supplied
###   with centring of the grid controlled by the keyword <centered>.
###

function get_grid(vectors;a0=nothing,E_cut=nothing,scale=1.0,centered=true)

    ##  Use grid spacing a0 if it is supplied, otherwise use any plane-wave 
    ##  cutoff to get a0.
    ##
    if (a0 != nothing)
        a0=a0
    elseif (E_cut != nothing)
        a0 = pw_to_dpw_cutoff(E_cut,dpw_units="bohr",scale=scale)
    end

    ##  Find the number of points that can be placed with spacing <a0>
    ##  along the vectors.  Generate the grid correspondong to this.
    ##
    npts            =  get_npts(vectors,a0)
    grid            =  GridData(npts, vectors; centered=true);

    ##  Build distace tables for the grid coordinates and their physical
    ##  counterparts.
    ##
    grid.dtable     =  build_distance_table(npts);
    grid.dtable_rs  =  calc_distances_physical(grid.dtable,grid.spacing);

    return(grid)
end





###
###  FUNCTION build_dist_table(grid)
### ======================================================================
###   
###   Build a table of distances between lattice points.  We work in
###   reduced (dimensionless) units.
###
function build_distance_table(npts::Vector{Int64})
    
    v0 = [0,0,0]

    ##  Update to use a more well-motivated set of reflections
    ##  Currently arbitrary yet sufficient.
    ##
    ##  reflectors = [[-1,1,1],[1,-1,1],[1,1,-1],[-1,0,0],[0,-1,0],[0,0,-1]]
    
    reflectors = [[-1,1,1],[1,-1,1],[1,1,-1]]

    tmp1 = []
    tmp2 = []

    dvec_set = []
    

    sp        = sortperm(npts)
    npts_sort = npts[sp]

    # gset = [[i,j,k] for i in 0:(npts[1]-1) 
    #                     for j in i:(npts[2]-1) 
    #                         for k in j:(npts[3]-1)]


    gset = [[i,j,k] for i in 0:(npts_sort[1]-1) 
                        for j in i:(npts_sort[2]-1) 
                            for k in j:(npts_sort[3]-1)]

    ##  Build displacements for all permutations of coordinates in one 
    ##  corner "wedge" of the grid.
    ##
    println("\tBuilding grid distance permutations: ")
    flush(stdout)
    @showprogress for gv in gset
        for vec in collect(unique(permutations(gv)))
            push!(tmp1,v0 - vec)
        end
    end
    
    ##  Apply symmetry operations (reflections) to generate displacements 
    ##  that are not produced by the perturbations above.
    ##
    println("\tApplying symmetry operations: ")
    flush(stdout)
    @showprogress for vec in tmp1
        for idx=1:length(reflectors)
            newvec = vec .* reflectors[idx]
            c1 = ( !(newvec in tmp1)  &&  !(-newvec in tmp1) )
            c2 = (  (newvec in tmp2)  ||   (-newvec in tmp2) )
            if ( c1 && (!c2) )
                push!(tmp2,newvec)
            end
        end
    end

    dvec_set = append!(tmp1[1:end],tmp2)

    ndv = length(dvec_set)

    println("\tGenerated a minimal set of ",ndv," displacement vectors.")

    udict = unique_distance_dict(dvec_set)
    n_dist = length(udict)

    println("\tGenerated a minimal set of ",n_dist," distances.\n")

    return(dvec_set)

end





###
###  FUNCTION make_dtable_dict(grid)
### ======================================================================
###   
###  Build a dict that maps every coordinate in the distance table from
###  <grid> to a scalar index.
###

function make_dtable_dict(grid)

    dtable_dict_p      =  Dict( grid.dtable[i] => i for i = 1:length(grid.dtable))
    dtable_dict_m      =  Dict(-grid.dtable[i] => i for i = 1:length(grid.dtable))

    grid.dtable_dict   =  merge(dtable_dict_p,dtable_dict_m);

end





###
###  FUNCTION make_dtable_dict(dtable)
### ======================================================================
###   
###  Build a dict containing unique distances in the table <dtable>.  
###  Values are rounded to <round_prec> before determining equivalence.
###  A vector of grid spacings <avec> can be used to build the resulting
###  dict in physical units.
###

function unique_distance_dict(dtable,;round_prec=10,avec=[1,1,1])

    ddict = Dict()

    d_id = 1

    for disp in dtable
        d = round(norm(disp .* avec),digits=round_prec)
        if !(d in keys(ddict))
            ddict[d]  =  d_id
            d_id     +=  1
        end
    end

    return(ddict)

end





###
###  FUNCTION build_distance_table(dtable)
### ======================================================================
###   
###  Build a distance table for a grid with <npts> with (physical)
###  grid spacings given by the array <avec>.
###

function build_distance_table(npts::Vector{Int64},avec::Vector{Float64})

    dvec_set_dist = []

    dvec_set = build_distance_table(npts)

    for n=1:length(dvec_set)
        push!(dvec_set_dist, dvec_set[n] .* avec)
    end

    return(dvec_set_dist)

end





###
###  FUNCTION calc_distances_physical(dvec_set,avec)
### ======================================================================
###  
###  Transfer the coordinates / displacements in raw grid units into
###  physical distances, as fixed by the grid spacings in the vector
###  <avec>.
###

function calc_distances_reciprocal(dvec_set::Any,avec::Vector{Float64})

    dvec_set_dist = []

    for n=1:length(dvec_set)
        push!(dvec_set_dist, dvec_set[n] .* avec)
    end

    return(dvec_set_dist)

end





###
###  FUNCTION calc_distances_physical(dvec_set,avec)
### ======================================================================
###  
###  Transfer the coordinates / displacements in raw grid units into
###  physical distances, as fixed by the grid spacings in the vector
###  <avec>.
###

function calc_distances_physical(dvec_set::Any,avec::Vector{Float64})

    dvec_set_dist = []

    for n=1:length(dvec_set)
        push!(dvec_set_dist, dvec_set[n] .* avec)
    end

    return(dvec_set_dist)

end







###
###  FUNCTION calc_grid_spacing(vectors,npts)
### ======================================================================
###  
###  Calculate the spacing between points for a grid defined by vectors
###  <vectors> with <npts> along each of the axes.
###

function calc_grid_spacing(vectors,npts)
    return( diag(hcat((vectors ./ npts)...)) )
end