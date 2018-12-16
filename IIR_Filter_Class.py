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

"""
# Normalised frequency 0.1
# T=1
f = 0.1

# Q factor
q = 10

# s infinity as defined for a 2nd order resonator (see impulse invariance)
si = np.complex(-np.pi*f/q,np.pi*f*np.sqrt(4-(1/(q**2))))

b0 = 1
b1 = -1
a1 = np.real(-(np.exp(si)+np.exp(np.conj(si))))
a2 = np.exp(2*np.real(si))

f = IIR2filter(b0,b1,a1,a2)

x = np.zeros(100)
x[10] = 1
y = np.zeros(100)

# Create the filter here
for i in range(len(x)):
    y[i] = f.filter(x[i])

plt.plot(y)

# Matched Z transform
w = np.arange(0,np.pi,0.01) # w is omega and 0.01 is steps (make it decent)
z = np.exp(1j*w)

h = (1-z**-1)/(1 - (np.exp(si)+np.exp(np.conj(si)))*z**-1 + np.exp(2*np.real(si))*z**-2)

plt.plot(np.linspace(0,0.5,len(h)),np.abs(h)) # Matched Z transform
"""