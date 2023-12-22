"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2023 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""


################################################################################
###                                                                          ###
###    Operator Lattice - Routines to manipulate operator-tagged lattices    ###
###                                                                          ###
################################################################################


import  numpy                    as      np
import  networkx                 as      netx

from    copy                             import  deepcopy
from    pyLIQTR.clam.operator_strings    import  op_strings

from    pyLIQTR.clam.cell_complex        import  OperatorCell
from    pyLIQTR.clam.operator_complex    import  Site, Edge, Plaquette

from    pyLIQTR.clam.utilities           import  tupleize_sites

from    pyLIQTR.clam.lattice             import  Lattice, UnitCell




##  OperatorUnitCell()                             (UnitCell)           
## -------------------------------------------------------------
##
##  UnitCell decorated with operators and coefficients.  
## 
##  Arguments:
##
##      td         -      Term dictionary for on-site and bond
##                        operators (keys are site type labels)
##
##                          td  =  {}
##                          td['A']  =  [ [ 'X', 'Z' ], [ 1.0, 0.5 ]]
##                          td['AA'] =  [ [ 'ZZ' ], [ -1.0 ] ]
##
##                        The '_' keyword will apppy the associated terms 
##                        to all sites and edges without a td entry.
##                         
##      sites      -      List of Sites in unit cell
##      edges      -      List of Edges in unit cell
##      dummy_sites  -    Dummy sites used to identify terms outside
##                        of the unit cell in the periodic system.
##      asymmetric_ops -  Whether term keys are treated as symmetric.  For
##                        instance, 'AB' and 'BA' have the same term by
##                        by default (i.e., only one term must appear in the
##                        dictionary <td>).  The asymmetric case requires 
##                        different entries for 'AB' and 'BA'.
##      double         -  Creates bipartite lattice by duplicating all sites
##                        and edges in the input.  The lattices are linked
##                        by edge terms specified in the <onsite=> dict. 
##      vectors        -  Array of cell vectors (for building repeats in an
##                        OperatorLattice).
##      

class OperatorUnitCell(UnitCell):

    def __init__( self, td, 
                        sites=None, 
                        dummy_sites=None, 
                        edges=None, 
                        asymmetric_ops=False, 
                        double=False,
                        onsite=None,
                        vectors=None   ):

        
        self.site_set = []
        self.edge_set = []

        self.plaquettes  =  []

        self._asymmetric_ops = asymmetric_ops

        self.td             =  td
        self.td_onsite      =  onsite
        self.__sites        =  sites
        self.__edges        =  edges
        self.__dummy_sites  =  dummy_sites
        self.vectors        =  vectors

        self.double         =  double

        self.build_cell()

        include_sites = self.site_set + self.edge_set

        UnitCell.__init__(self, sites=include_sites)

        if self.td_onsite is not None:
            self.double = True

        if self.double:
            self.__double()



    ## Get the term dictionary entry associated
    ## with label <label>.
    ##
    def get_term(self,label,term_dict=None):

        if (term_dict is None):
            term_dict = self.td

        if not self._asymmetric_ops:

            if label in term_dict.keys():
                term_label  =  label
            elif label[::-1] in term_dict.keys():
                term_label  =  label[::-1]
            elif ('_' in term_dict):
                term_label  =  '_'
            else:
                raise Exception("Term ", label, " not found in term dictionary.")
                # should just set this to zero in actuality        

 #       return(term_dict[term_label])
        return(term_label)


    ## Populate the unit cell using the input
    ##
    def build_cell(self):

        for site in self.__sites.keys():
            self.add_site(site,self.__sites[site])

        for edge in self.__edges:

            label = ""
            for site in edge:

                if (site in self.__sites):
                    label += self.__sites[site]
                elif (site in self.__dummy_sites):
                    label += self.__dummy_sites[site]

            term_label = self.get_term(label)

            # if not self._asymmetric_ops:
            #     if label in self.td.keys():
            #         pass
            #     elif label[::-1] in self.td.keys():
            #         label = label[::-1]
            #     else:
            #         raise Exception("Term ", label, " not found in term dictionary.")
            #         # should just set this to zero in actuality

            self.add_edge(edge[0],edge[1],term_label)
        


    ## Wrapper to add a site <s> to the cell, with operators / coefficients
    ## extrated using label <label> from the term dictionary.
    ##
    def add_site(self, s, label):
        site = Site(s, ops=self.td[label][0], cons=self.td[label][1], 
                      label=label)
        self.site_set.append(site)
        return


    ## Wrapper to add a edge <s0>, <s1> to the cell, with operators / coefficients
    ## extrated using label <label> from the term dictionary.
    ##
    def add_edge(self, s0, s1, label):
        edge = Edge(s0, s1, ops=self.td[label][0], cons=self.td[label][1],
                      label=label)
        self.edge_set.append(edge)
        return



    ## Double the unit cell into bipartitie sublattices, and couple these 
    ## lattices with the terms specified from <self.td_onsite>
    ##
    def __double(self):

        __sites  =  self.skeletons[0]
        __edges  =  self.skeletons[1]
        
        n_sites = len(__sites)

        # if (len(onsite_terms) < n_sites):
        #     onsite_terms = [onsite_terms[0]]*n_sites

        sites_new   =  []
        edges_new   =  []
        edges_bond  =  []

        vectors_new =  []

        for site in __sites:

            b_even,b_odd = self.__double_basis(site.basis[0])

            s_even = Site(b_even, ops=site.ops(), cons=site.cons(),
                            label=site.label)
            s_odd  = Site(b_odd,  ops=site.ops(), cons=site.cons(), 
                            label=site.label+"_d")

            sites_new.append(s_even)
            sites_new.append(s_odd)

            term = self.get_term(site.label,term_dict=self.td_onsite)


            e_double = Edge(b_even,b_odd, ops=self.td_onsite[term][0], 
                            cons=self.td_onsite[term][1],label=site.label+"_d")

            edges_bond.append(e_double)

        for edge in __edges:

            b0_even,b0_odd = self.__double_basis(edge.basis[0])
            b1_even,b1_odd = self.__double_basis(edge.basis[1])

            e_even = Edge(b0_even,b1_even, ops=edge.ops(), cons=edge.cons(), 
                            label=edge.label)
            e_odd  = Edge(b0_odd,b1_odd, ops=edge.ops(), cons=edge.cons(), 
                            label=edge.label+"_d")

            edges_new.append(e_even)
            edges_new.append(e_odd)

        for vec in self.vectors:
            vec_new  =  deepcopy(vec)
            vec_new[-1] = 2*vec_new[-1]
            vectors_new.append(vec_new)

        self._doubled = True

        self.skeletons[0] = []
        self.skeletons[1] = []

        for site in sites_new:
            self.add(site)

        for edge in (edges_new + edges_bond):
            self.add(edge)

        self.vectors = vectors_new    



    ## Double a given basis index into sites on even
    ## and odd index sublattices.
    ##
    def __double_basis(self,basis):

        d = len(basis)

        basis_even  =  []
        basis_odd   =  []

        for n in range(d-1):

            basis_even.append(basis[n])
            basis_odd.append(basis[n])

        basis_even.append( 2*basis[-1])
        basis_odd.append(  2*basis[-1] + 1)

        return(basis_even,basis_odd)







##  OperatorLattice()                             (Lattice)           
## -------------------------------------------------------------
##
##  Lattice built up from an OperatorUnitCell object.
## 
##  Arguments:
##
##      size       -      Tuple specifying lattice dimensions in 
##                        terms of unit cell repeats.
##      unit       -      Unit cell to be repeated in lattice
##                        (of type OperatorUnitCell)
##      vectors    -      Lattice vectors, if not supplied via entry
##                        in OperatorUnitCell.
##      periodic   -      Tuple indicating if the lattice is periodic 
##                        in each dimension.
##      type       -      Type of indexing scheme:
##
##                          'op-lattice'    -    indexing with native site indicies
##                          'op-qubit'      -    indexing via qubit labeling strategy
##
##      strategy   -      Indexing strategy when mapping latice sites to qubits
##                        (for type='op-qubit').  Arguments
##
##                          'zigzag'        -    Zigzag path through lattice
##                          'armchair'      -    Armchair path through lattice
##                          'custom'        -    Custom index function passed as 
##                                               < custom_index_function = >
##
##      overhang   -      Rule for handling edges that dangle outside the target
##                        lattice:
##
##                          'ignore'        -   retain hanging terms even if no lattice
##                                              site is present.
##                          'drop'          -   drop hanging terms
##                          'fill'          -   add in sites that would be linked by these
##                                              edges by referencing back to the unit cell.
##                                              ( *** not implemented *** )
##            
##      vectors        -  Array of cell vectors (for building repeats in an
##                        OperatorLattice).
##  

class OperatorLattice(Lattice):
    
    def __init__(self,  size=None,
                        unit=None,
                        vectors=None,  
                        periodic=None,
                        type='op-qubit', 
                        strategy='zigzag',
                        overhang='drop',
                        custom_index_function=None   ):

        self.type      =  type  
        self.op_set    =  []
        self.strategy  =  strategy

        self.custom_index_function = custom_index_function

        if (vectors is None):
            vectors = unit.vectors

        Lattice.__init__(self,vectors,size=size,unit=unit,periodic=periodic)

        if (size is not None):
            self.build(size,overhang=overhang)


    ## Return operator strings corresponding to lattice, with indexing given
    ## by <type> {'op-lattice', 'op-qubit'}.  Indexing strategy <strategy>
    ## may be supplied when <type='op-qubit'>
    ##
    def op_strings(self,type=None,strategy=None):

        if strategy is None:
            strategy = self.strategy

        op_str = self.lattice_to_ops(strategy=strategy,type=type,
                                    return_as='op_strings')
        return(op_str)
    


    ## Return list of operators from lattice.  See op_strings() for
    ## arguments.
    ##
    def operator_list(self,type=None,strategy=None,with_label=False):

        if strategy is None:
            strategy = self.strategy

        op_list = self.lattice_to_ops(strategy=strategy,type=type,
                                        return_as='list',with_label=with_label)
        return(op_list)



    ## Handles conversion of lattice terms into formats that can be ingested 
    ## by other routines.
    ##
    def lattice_to_ops(self, strategy=None, 
                             type=None, 
                             custom_index_function=None, 
                             return_as='list', 
                             with_label=False  ):

        if strategy is None:
            strategy = self.strategy

        if (type is None):
            type = self.type

        if (return_as == 'list'):
            op_set = []
        elif (return_as == 'op_strings'):
            op_set = op_strings()
            op_set.set_num_qb(len(self.skeletons[0]))  # set number of qubits
        elif (return_as == 'tensor_product'):
            op_set = TensorOperator()

        lindex = self.__handle_index_set(strategy)

        for n in range(len(self.skeletons)):
            
            for basis,props,labels in zip(self.skeletons[n],self.props[n],
                                        self.labels[n]):

                for k in range(len(props)):
                    if (type == 'op-qubit'):

                        sites        =  []
                        reject_flag  =  False

                        for item in tupleize_sites(basis): 
                            if (item not in lindex.keys()):

                                reject_flag = True
                                continue
                            
                            sites.append(lindex[item])


                    elif (type == 'op-lattice'):
                        reject_flag = False
                        sites = tuple([item for item in tupleize_sites(basis)])

                    if (reject_flag):
                        continue

                    if (return_as == 'list'):
                        
                        if (isinstance(props[k][0],str)):
                            op_entry = op_tuple_to_string(tuple(props[k][0]))
                        else:
                            op_entry = props[k][0]

                        if (with_label):
                            pack = [sites, op_entry, props[k][1], labels]
                        else:
                            pack = [sites, op_entry, props[k][1]]

                        op_set.append(pack)
                        
                    elif (return_as == 'op_strings'):
                        op_set.append(sites, op_tuple_to_string(tuple(props[k][0])), 
                                        props[k][1])

        return(op_set)




    def __handle_index_set(self,strategy):

        if   ( strategy == 'zigzag' ):
            lindex = get_index_set_zigzag(self.skeletons[0])
        elif ( strategy == 'snake' ):
            lindex = get_index_set_snake(self.skeletons[0])
        elif (strategy == 'custom'):
            if (self.custom_index_function is not None):
                lindex = get_index_set_snake(self.skeletons[0])
            else:
                raise Exception("Custom lattice-to-qubit indexing function not supplied "+
                                "(use keyword custom_index_function= ).")
        else:
            raise Exception("Unsupported index strategy: ",str(strategy))
        

        return(lindex)
        







##  get_index_set_zigzag()                                     
## -------------------------------------------------------------
##
##  Index path that zigzags through lattice based on sites in
##  list <sites>.  Assumes (row, col, z) order.
##  

def get_index_set_zigzag(sites):
  
    # assumes row, col, z order

    N    =  len(sites)
    dim  =  len(sites[0][0])

    idx_dict = {}
 
    site_array = sites_to_array(sites)

    if (dim == 2):
        site_array = site_array[np.lexsort((site_array[:,1], 
                                    site_array[:,0]))]
    elif (dim == 3):
        site_array = site_array[np.lexsort((site_array[:,2],site_array[:,1], 
                                    site_array[:,0]))]


    for n in range(len(site_array)):
        idx_dict[tuple(site_array[n])] = n

    return(idx_dict)






##  sites_to_array()                                     
## -------------------------------------------------------------
##
##  Builds an array of coordinate tuples from Site array <sites>
##  


def sites_to_array(sites):
    
    n_sites = len(sites)
    d_site  = len(sites[0][0])
    site_array = np.zeros((n_sites,d_site))

    for idx in range(n_sites):
        site_array[idx] = sites[idx][0]

    return(site_array)





##  op_tuple_to_string()                                     
## -------------------------------------------------------------
##
##  Concatenates a tuple of operator labels into a string
##  

def op_tuple_to_string(ops):
    string = ''
    for op in ops:
        string += op
    return(string)





##  draw_lattice_2d()                                     
## -------------------------------------------------------------
##
##  Draw a 2d projection of the OperatorLattice object <LT>.  
##
##  Arguments:
##
##      LT               -      OperatorLattice object
##      single_spin      -      Plot only a single sublattice
##                              of a doubled lattice
##      doubled          -      Use custom color scheme for doubled
##                              lattice (buggy, not recommended)
##      layout           -      Strategy for drawing lattice:
##
##                                  'spring'
##                                  'kamada_kawai'
##
##      node_size        -      Relative size of drawn nodes
##      with_labels      -      Option to print node labels on graph
##      font_weight      -      Font weight for node labels
##      label_type       -      Type of labels to use
##  

def draw_lattice_2d( LT,
                     color_nodes=True,
                     node_size=50,
                     font_weight='bold',
                     with_labels=False,
                     label_type='index',
                     layout='kamada_kawai',
                     doubled=False,
                     single_spin=False ):

    labels     =  {}

    finished   =  []
    edge_set   =  []

    op_strings = LT.op_strings()

    for term in LT.operator_list(with_label=True):

        if (len(term[0]) == 1):
            if ( term[0][0] not in finished ):
                    labels[term[0][0]] =  term[3]

        if (len(term[0]) == 2):
            if single_spin:
                if ( (term[3][-2:] != "_d") ):
                    edge_set.append(term[0])
            else:
                edge_set.append(term[0])

    G = netx.Graph()
    G.add_edges_from(edge_set)

    if (layout == 'kamada_kawai'):
        layout_manager = netx.kamada_kawai_layout(G,scale=3.0)
    elif (layout == 'spring'):
        layout_manager = netx.spring_layout(G,scale=3.0)


    if (color_nodes):

        if doubled:
            node_pallete  = {   0: 'cornflowerblue',  
                                1: 'blue', 
                                2: 'salmon', 
                                3: 'red', 
                                4: 'yellowgreen',  
                                5: 'green'             }
        else:
            node_pallete  = {   0: 'cornflowerblue', 
                                1: 'salmon', 
                                2: 'yellowgreen', 
                                3: 'blue', 
                                4: 'red', 
                                5: 'green'              }

        color_map     =  []
        unique_nodes  =  {}

        nidx = 0

        for node in G:

            if not doubled:
                if labels[node] not in unique_nodes.keys():
                    unique_nodes[labels[node]] = nidx
                    nidx += 1
                color_map.append(node_pallete[unique_nodes[labels[node]]])

            if doubled:
                if (labels[node][-2:] != "_d"):
                    if labels[node] not in unique_nodes.keys():

                        unique_nodes[labels[node]] = nidx
                        nidx += 1

                        unique_nodes[labels[node]+"_d"] = nidx
                        nidx += 1

                    color_map.append(node_pallete[unique_nodes[labels[node]]])
                    color_map.append(node_pallete[unique_nodes[labels[node]+"_d"]])

        netx.draw( G, with_labels=with_labels, font_weight=font_weight, 
                      pos=layout_manager, node_color=color_map, 
                      node_size=node_size, alpha=1.0, width=0.75, 
                      edge_color='black', style='--'  )
    else:
        netx.draw( G, with_labels=with_labels, font_weight=font_weight, 
                      pos=layout_manager, node_size=node_size, alpha=1.0, 
                      width=0.75, edge_color='black', style='--'  )
