import math


class Q:

        def __init__(self, n, u, v, coalRate):
                self.rows = ((n+1)*(n+1) + n - 1)/2
                self.col = ((n+1)*(n+1) + n - 1)/2
                self.n = n
                self.u = u
                self.v = v
                self.coalRate = coalRate

                #calculate the infinity normal of the Q matrix
                if(u>v):
                        self.inf = math.max(2.0*u*(n-1) + coalRate*(n-1)*(n-1), 2.0*u*(n) + coalRate*(n)*(n-1)/2.0)
                else:
                        self.inf = math.max(2.0*v*(n-1) + coalRate*(n-1)*(n-1), 2.0*v*(n) + coalRate*(n)*(n-1)/2.0)
                
                #calculate the trace
                self.trace = -coalRate*(n-1)*n*(n+1)*(n+2)/8 - n*(n+1)*(n+2)*(u+v)/6

        
        

