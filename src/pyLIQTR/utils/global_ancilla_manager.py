from cirq.ops.qubit_manager import QubitManager, CleanQubit, BorrowableQubit
from typing import Iterable, List, Set, TYPE_CHECKING

class GlobalQubitManager(QubitManager):
    def __init__(self, prefix: str = 'gancilla'):
        self._clean_set = set()
        self._dirty_set = set()
        self._inuse_set = set()
        self._locked_qubits = dict()
        self._clean_id = 0
        self._dirty_id = 0
        self._prefix = prefix
    
    def reset(self):
        self._clean_set = set()
        self._dirty_set = set()
        self._inuse_set = set()
        self._clean_id = 0
        self._dirty_id = 0

    def __repr__(self) -> str:
        outStr = f"GAM : {self._prefix}\n"
        outStr += "Clean set\n\t"
        outStr += str(self._clean_set)
        outStr += "\nDirty set\n\t"
        outStr += str(self._dirty_set)
        outStr += "\nInuse set\n\t"
        outStr += str(self._inuse_set)
        outStr += "\nLocked set\n\t"
        outStr += str(self._locked_qubits)
        outStr += "\n"
        return outStr

    def reserve_qubits(self, qubits : List['cirq.Qid'], level:int):
        if level not in self._locked_qubits:
            self._locked_qubits[level] = set()

        for q in qubits:
            if isinstance(q, CleanQubit):
                self._locked_qubits[level].add(('clean',q))
            elif isinstance(q, BorrowableQubit):
                self._locked_qubits[level].add(('dirty',q))
    
    def unlock_qubits(self, level:int):
        q2free = list(self._locked_qubits[level])
        self._locked_qubits[level] = set()
        for (type,q) in q2free:
            #make sure the qubit isnt in another level. only the parent can fully clear.
            doAllow = True
            for lvl,lqs in self._locked_qubits.items():
                if lvl == level:
                    continue
                tmp = set([q[1] for q in lqs])
                if q in tmp:
                    doAllow=False
                    break
            if not doAllow:
                continue
            if type == 'clean':
                self._clean_set.add(q)
            elif type == 'dirty':
                self._dirty_set.add(q)
            if q in self._inuse_set:
                self._inuse_set.remove(q)

    def qalloc(self, n: int, dim: int = 2) -> List['cirq.Qid']:
        if len(self._clean_set) < n:
            #need to allocate new qubits
            #need to grow our clean set.
            self._clean_id += n
            newqubits = [CleanQubit(idx, dim, self._prefix) for idx in range(self._clean_id-n, self._clean_id)]
            self._clean_set = self._clean_set.union(set(newqubits))
        else:
            #can allocate directly from our clean set, dont need to do anything
            pass

        #grab the qubits we are interested in using.
        qubits2use = [self._clean_set.pop() for __ in range(n)]
        self._inuse_set = self._inuse_set.union(set(qubits2use))
        return qubits2use

    def qborrow(self, n: int, dim: int = 2) -> List['cirq.Qid']:
        if len(self._dirty_set) < n:
            #need to allocate new dirty qubits
            self._dirty_id += n
            newqubits = [ BorrowableQubit(i, dim, self._prefix) for i in range(self._dirty_id - n, self._dirty_id)]
            self._dirty_set = self._dirty_set.union(set(newqubits))
        else:
            #can allocate directly, need nothing.
            pass

        qubits2use = [self._dirty_set.pop() for __ in range(n)]
        self._inuse_set = self._inuse_set.union(set(qubits2use))
        return qubits2use

    def qfree(self, qubits: Iterable['cirq.Qid']) -> None:
        keep_locked_qubits = set()
        for __,lq in self._locked_qubits.items():
            tmp = set([q[1] for q in lq])
            keep_locked_qubits = keep_locked_qubits.union(tmp)

        for q in qubits:
            if q in keep_locked_qubits:
                continue
                
            assert(q in self._inuse_set)
            if isinstance(q,CleanQubit):
                assert(q.id < self._clean_id)
                self._clean_set.add(q)
                self._inuse_set.remove(q)
            elif isinstance(q,BorrowableQubit):
                assert(q.id < self._dirty_id)
                self._dirty_set.add(q)
                self._inuse_set.remove(q)
            else:
                raise ValueError("Unknown qubit type???")    

gam = GlobalQubitManager()