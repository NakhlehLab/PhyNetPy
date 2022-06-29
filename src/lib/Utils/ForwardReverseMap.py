class biMap(dict):


        def __init__(self, *args, **kwargs):
                super(biMap, self).__init__(*args, **kwargs)
                self.inverse = {}
                for key, value in self.items():
                        self.inverse.setdefault(value,[]).append(key) 

        def __setitem__(self, key, value):
                if key in self:
                        self.inverse[self[key]].remove(key) 
                super(biMap, self).__setitem__(key, value)
                self.inverse.setdefault(value,[]).append(key)        

        def __delitem__(self, key):
                self.inverse.setdefault(self[key],[]).remove(key)
                if self[key] in self.inverse and not self.inverse[self[key]]: 
                        del self.inverse[self[key]]
                super(biMap, self).__delitem__(key)

        def isInjective(self):
                for value in self.values():
                        if len(self.inverse[value])>1:
                                return False
                return True





