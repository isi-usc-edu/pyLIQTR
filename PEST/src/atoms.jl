# Copyright (c) 2023 Massachusetts Institute of Technology 
# SPDX-License-Identifier: MIT


################################################################################
###                                                                          ### 
###           ATOMS  -  Coordinate IO routines for pyLIQTR's PEST            ###         
###                                                                          ### 
################################################################################



###
###  DICT   dt_atomic_numbers
### ============================================================================
###   
###   Dict containing the atomic number for each element (standard 
###   abbreviations as the key).  These should be right, but it was
###   based on a copy-and-paste job.
###

dt_atomic_numbers =    Dict( "H" =>   1,   "He" =>  2,   "Li" =>  3,   "Be" =>  4,   
 "B" =>   5,   "C" =>   6,   "N" =>   7,   "O" =>   8,   "F" =>   9,   "Ne" => 10,  
 "Na" => 11,   "Mg" => 12,   "Al" => 13,   "Si" => 14,   "P" =>  15,   "S" =>  16,   
 "Cl" => 17,   "Ar" => 18,   "K" =>  19,   "Ca" => 20,   "Sc" => 21,   "Ti" => 22,
 "V" =>  23,   "Cr" => 24,   "Mn" => 25,   "Fe" => 26,   "Co" => 27,   "Ni" => 28,
 "Cu" => 29,   "Zn" => 30,   "Ga" => 31,   "Ge" => 32,   "As" => 33,   "Se" => 34,   
 "Br" => 35,   "Kr" => 36,   "Rb" => 37,   "Sr" => 38,   "Y" =>  39,   "Zr" => 40,  
 "Nb" => 41,   "Mo" => 42,   "Tc" => 43,   "Ru" => 44,   "Rh" => 45,   "Pd" => 46,   
 "Ag" => 47,   "Cd" => 48,   "In" => 49,   "Sn" => 50,   "Sb" => 51,   "Te" => 52,   
 "I" =>  53,   "Xe" => 54,   "Cs" => 55,   "Ba" => 56,   "La" => 57,   "Ce" => 58,   
 "Pr" => 59,   "Nd" => 60,   "Pm" => 61,   "Sm" => 62,   "Eu" => 63,   "Gd" => 64,   
 "Tb" => 65,   "Dy" => 66,   "Ho" => 67,   "Er" => 68,   "Tm" => 69,   "Yb" => 70,
 "Lu" => 71,   "Hf" => 72,   "Ta" => 73,   "W" =>  74,   "Re" => 75,   "Os" => 76,   
 "Ir" => 77,   "Pt" => 78,   "Au" => 79,   "Hg" => 80,   "Tl" => 81,   "Pb" => 82,   
 "Bi" => 83,   "Po" => 84,   "At" => 85,   "Rn" => 86,   "Fr" => 87,   "Ra" => 88,   
 "Ac" => 89,   "Th" => 90,   "Pa" => 91,   "U" =>  92,   "Np" => 93,   "Pu" => 94,   
 "Am" => 95,   "Cm" => 96,   "Bk" => 97,   "Cf" => 98,   "Es" => 99,   "Fm" => 100,  
 "Md" => 101,  "No" => 102,  "Lr" => 103    );






###
###  STRUCT   Atoms(filename;recenter=true,convert_to_bohr=true)
### ============================================================================
###   
###   Container to hold atomic coordinates. Data are read from xyz file
###   <filename> and centered about the origin contingent on the flag
###   <recenter>. Input units are converted from angstrom to Bohr if 
###   <convert_to_bohr> is true.
###

mutable struct Atoms 

    n_atom    :: Int64            #  Number of atoms in geometry
    atnum     :: Any              #  Atomic numbers for each atom (matches coords)
    coords    :: Any              #  Cartesian coordinates for each atom
    lengths   :: Any              #  Lengths of enclosing cell.
    vectors   :: Any
    bounds    :: Any
    center    :: Any

    function Atoms(input;recenter=true,convert_to_bohr=true)

        ## Load coordinates from file or load array into container
        ##
        if (typeof(input) == String)
            atnum, coords = load_xyz(input)
        elseif (typeof(input) == Array{Float64, 2})
            atnum  = input[:,1]
            coords = input[:,2:4] 
        end
        
        ##  Get number of atoms, maximal extent of rectangular cell enclosing
        ##  atoms, and lattice vectors for this cel.
        ##
        n_atom    =  length(atnum)
        minmax_1  =  minmax(coords)
    
        lengths   =  vec(sum(abs.(minmax(coords)),dims=2))
        vmat      =  collect(Diagonal(lengths))
        vectors   =  [vmat[1,:], vmat[2,:], vmat[3,:]]
 

        ##  Center atoms at origin if requested (default)
        ##
        if (recenter)
            shift = zeros(3)
            for n=1:3
                shift[n] = lengths[n]/2 - minmax_1[n,2]
            end
            for n=1:n_atom
                coords[n,:] += shift
            end

        end

        ##  Convert units from angstrom to Bohr; save center and 
        ##  bounds
        convert_to_bohr && ( coords  = coords  * angstrom_to_bohr )
        convert_to_bohr && ( vectors = vectors * angstrom_to_bohr )
        convert_to_bohr && ( lengths = lengths * angstrom_to_bohr )

        bounds  =  minmax(coords)
        center  =  sum(bounds,dims=2)/2
        
        new(n_atom,atnum,coords,lengths,vectors,bounds,center)

    end
end





###
###  FUNCTION   atoms, vectors = get_geometry()
### ======================================================================
###   
###   Wrapper to load geometry and add padding. The geometry is returned as 
###   at Atoms container, and the basis vectors are padded by <pad> if 
###   specified (we assume that padding is passed in angstroms by default).  
###   The vectors are returned in atomic units (Bohr).
###
###   If <pad_to_bohr> is true, the <pad> will be converted to Bohr.  If
###   fase, it is assumed that they have been passed in Bohr.
###

function get_geometry(filename::String,;pad=[0.0,0.0,0.0],pad_to_bohr=true)

    ##  Load geometry from file
    ##
    mol       =  load_coordinates(filename)    

    ##  Convert padding from angstrom to Bohr (default)
    ##
    if pad_to_bohr
        pad_bohr  =  pad*angstrom_to_bohr
    else
        pad_bohr  =  pad
    end

    ##  Add padding to basis vectors
    ##
    vectors = pad_vectors(mol.vectors,pad=pad_bohr)

    return (mol,vectors)
end





###
###  FUNCTION   npts = get_npts(vectors,a0)
### ======================================================================
###   
###   Get number of basis functions along each axis, assuming a regular, 
###   isotropic grid with spacing a0.  Rounds up.
###

function get_npts(vectors,a0)
    npts = zeros(Int,3)
    for n=1:3
        npts[n] = Int(ceil(vectors[n][n]/a0))
    end
    return (npts)
end





###
###  FUNCTION   coords = load_coordinates(coords::Atoms)
###  FUNCTION   coords = load_coordinates(file_str::String)
### ======================================================================
###   
###   Multiple dispactch wrappers for loading coordinates. Passes an Atoms
###   container back if it is given as an argument; if a string is given 
###   it will be treated as a filename (and loaded into an Atoms container).
###

function load_coordinates(coords::Atoms)
    return(coords)
end

function load_coordinates(file_str::String)
    return(Atoms(file_str))
end





###
###  FUNCTION   atnum, coords = load_xyz(filename)
### ============================================================================
###   
###   Load atomic coordinates from xyz format file <filename>.  Returns an 
###   arrays <atnum> containing the atomic number a given atom and and 
###   <coords> containing the cartesian coordinates.  No unit conversion 
###   is performed so the resulting units match the input file.
###
###   NOTE:   Atom label must be uppercase
###

function load_xyz(filename)

    lines  =  readlines(filename)
    num    =  parse(Int64, lines[1])

    names  =  []

    coords =  zeros((num,3))
    atnum  =  zeros(Int64,num)

    ##  Extract arrays for atomic numbers and coordinates
    ##
    for n = 3:(3+num-1)
        fields = split(lines[n])
        atnum[n-2] = dt_atomic_numbers[string(fields[1])]
        crds = [parse(Float64,val) for val in fields[2:4]]
        for m = 1:3
            coords[n-2,m] = crds[m]
        end
    end
    
    return(atnum,coords)
end





###
###  FUNCTION   padded_vectors = pad_vectors(vectors)
### ======================================================================
###   
###   Add padding to basis vectors, assuming a rectangular coordinate 
###   system.  No unit conversion is performed.
###
###     TODO: Support arbitary paralellpiped cell by enlarging basis vectors
###           the specified distance by a vector parallel to each basis
###           element
###

function pad_vectors(vectors;pad=[0.0,0.0,0.0])

    pad_arr          =  [ [pad[1], 0.0, 0.0], [0.0, pad[2], 0.0], [0.0,0.0, pad[3]] ] 
    padded_vectors   =  (vectors + pad_arr) 

    return(padded_vectors)
end






