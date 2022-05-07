#!/usr/bin/env python3

import pyAgrum as gum, pymc3 as pm, theano,os
from tempfile import mkstemp

class BN:

    def __init__(self, bif_file):
        self.agrum:gum.BayesNet = gum.loadBN(bif_file)
        self.bif_string = open(bif_file).read()
        self.pymc3 = BN.agrum2PyMC3(self.agrum)

    def __getstate__(self):
        return self.bif_string, self.pymc3, self.getMetaData()

    def __setstate__(self, state):
        self.bif_string, self.pymc3, metadata = state
        # save bif_string to temp file
        fd, name = mkstemp(suffix='.bif')

        fp = open(fd, 'w')
        fp.write(self.bif_string)
        fp.close()

        self.agrum: gum.BayesNet = gum.loadBN(name)
        assert metadata == self.getMetaData()
        os.remove(name)

    def getVarNames(self):
        return self.agrum.names()

    def getValLabels(self, var):
        return list(self.agrum.variable(var).labels())

    def getVarId(self, var_name:str):
        return self.getVarNames().index(var_name)

    def getParentsId(self, var):
        return self.agrum.parents(var)

    def getParentsName(self, var):
        var_names = self.getVarNames()
        return [var_names[i] for i in self.agrum.parents(var)]

    def getValId(self,var,val_label:str):
        return self.agrum.variable(var).index(val_label)

    def getCPT(self, var):
        return self.agrum.cpt(var).toarray()

    def getMetaData(self):
        metadata = {'VarNames': self.getVarNames(),
                    'ValLabels': {vn: self.getValLabels(vn) for vn in self.getVarNames()}}
        return metadata

    def getMoralizedNeighborIds(self, var):
        if isinstance(var, str):
            var = self.getVarId(var)
        return self.agrum.moralGraph().neighbours(var)

    def getMoralizedNeighborsName(self, var):
        if isinstance(var, str):
            var = self.getVarId(var)
        return {self.getVarNames()[i] for i in self.agrum.moralGraph().neighbours(var)}

    @staticmethod
    def agrum2PyMC3(gum_bn: gum.BayesNet):
        index2var = gum_bn.names()  # number to variable name mapping
        topo_order = [index2var[i] for i in gum_bn.topologicalOrder()]
        assert len(index2var) == len(topo_order)

        m = pm.Model()
        pm_vars = {}

        with m as model:
            while len(pm_vars)<len(index2var):
                # we should use a topological sort
                for vn in index2var:
                    if vn in pm_vars:
                        continue

                    cpt_array = gum_bn.cpt(vn).toarray()
                    var = gum_bn.variableFromName(vn)

                    if len(gum_bn.parents(vn)) == 0:  #root nodes
                        pm_vars[vn] = pm.Categorical(vn, p=cpt_array)
                    else:
                        # you can only insert pm variables if their parents are already there

                        cpt_names = gum_bn.cpt(vn).var_names
                        assert cpt_names[-1]==vn
                        parent_names = cpt_names[:-1]
                        assert vn not in parent_names

                        if not set(pm_vars.keys()).issuperset(parent_names):
                            continue

                        p_vn = theano.shared(gum_bn.cpt(vn).toarray())

                        p_statement = 'p_vn['

                        for p in parent_names:
                            p_statement += 'pm_vars["{0}"],'.format(p)

                        p_statement=p_statement[:-1] + ']'

                        # print(p_statement)

                        p_tensor = eval(p_statement)

                        pm_vars[vn] = pm.Categorical(vn, p=p_tensor)

        return m



if __name__ == '__main__':

    import gzip,pickle,glob

    bif_files = glob.glob('models/*.bif')

    for f in bif_files:
        print(f)
        bn_obj = BN(f)
        new_fn = f.replace('.bif','.bn.gz')
        fp = gzip.open(new_fn, 'wb')
        pickle.dump(bn_obj, fp)
        fp.close()