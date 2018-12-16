""" IIR Filters: Task 3 """

class IIR2filter:
    def __init__(self,_a,_b):
        self.a0 = _a[0]
        self.a1 = _a[1]
        self.a2 = _a[2]
        
        self.b0 = _b[0]
        self.b1 = _b[1]
        self.b2 = _b[2]
        
        
        
        
    def filter(self,x):
        buffer1_in = 0 
        buffer2_in = 0
        buffer1_out = 0
        buffer2_out = 0
        acc = 0
        
        """FIR Part"""
        acc = x*self.b0 + buffer1_in*self.b1 + buffer2_in*self.b2
        
        """IIR Part"""
        acc = acc - buffer1_out*self.a1 - buffer2_out*self.a2
        buffer2_in = buffer1_in
        buffer1_in = x
        buffer2_out = buffer1_out
        buffer1_out = acc
        
        return acc
