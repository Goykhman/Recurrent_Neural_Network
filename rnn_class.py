'''
*******************************************************************************

Recurrent Neural Network with tanh() classifier for intput(S)->output(N)
training, inspired by the code by Andrej Karpathy, 

https://gist.github.com/karpathy/d4dee566867f8291f086

We will use adaptive learning rate method to control the learning.
The corresponding gradient norms will be kept in the gradient memory
arrays, for details of the method see
        
http://cs231n.github.io/neural-networks-3/#ada

All the vectors are saved as Nx1 arrays, so that we will be able to return
direct products using np.dot().

*******************************************************************************

The RNN gets the input sequence A of some (unspecified) size 's'
and the output sequence B of the same size 's'. We want to train
the RNN to predict subsequences of B of size T from the corresponding
subsequences of A of size T,
    
    A=A[1] ... A[p] ... A[p+T-1] ... A[s]
    B=B[1] ... B[p] ... B[p+T-1] ... B[s]
    
for arbitrary starting point p of the subsequences.

This corresponds to unrolling RNN for T time layers. We can choose
T independently of the size of A and B.

The input states A[t] take one of S values and are treated by the RNN
as S-vectors. The output states B[t] take one of N values and are
treated by the RNN as numbers in the range 0...N-1.

*******************************************************************************
'''

from time import time
import numpy as np
np.random.seed(int(time()))

class RNN:   
    def __init__(self,T,S,M,N,A,B,r):
        '''
        Attributes:            
            T -- number of time steps for which we unroll the RNN.            
            S -- number of the input states.            
            M -- size of the memory vectors.            
            N -- number of the output states.
            A -- input data, sequence of size 's'.
            B -- output data, sequence of size 's'.
            W -- input to memory matrix.            
            V -- memory to memory matrix.            
            U -- memory to output matrix.            
            b -- memory bias.            
            e -- output bias.  
            r -- learning rate.
        '''        
        self.T=T        
        self.S=S        
        self.M=M        
        self.N=N  
        self.A=A
        self.B=B
        self.W=0.01*np.random.randn(self.M,self.S)
        self.V=0.01*np.random.randn(self.M,self.M)
        self.U=0.01*np.random.randn(self.N,self.M)
        self.b=np.zeros((self.M,1))
        self.e=np.zeros((self.N,1))
        self.r=r                
        '''
        The original loss function is calculated for the
        probabilities of a uniform guess for each letter, 1.0/N,
        at each of T time steps.
        '''        
        self.Loss=-np.log(1.0/self.N)*self.T
        '''
        Gradients norms:
        '''        
        self.norms_W=np.zeros((self.M,self.S))
        self.norms_V=np.zeros((self.M,self.M))
        self.norms_U=np.zeros((self.N,self.M))
        self.norms_b=np.zeros((self.M,1))
        self.norms_e=np.zeros((self.N,1))
    def gradients(self,x,y,m_seed):
        '''
        Arguments:   
            x -- list of input S-vectors of size T.    
            y -- list of desired output numbers of size T.            
            m_seed -- seed memory '-1' layer for t=0.            
        Returns:            
            tuple: loss, gradients, (T-1)-memory (last) state.           
            loss is a number, gradients are returned in the
            order nabla_W, D_V, nabla_U, nabla_b, nabla_e and
            have the shape of their corresponding matrices.
        We will save time dependence of memory vectors (Mem),
        outputs (Y), and probability outputs (P) as dictionaries,
        where time will be a key. The pre-starting memory layer
        Mem[-1] is taken as m_seed, which is an argument to this
        function. Remember that Mem is a dictionary, so -1 means
        'key=-1', not the last element.  Also, m_seed is copied
        and will not be mutated by this function.
        The 'L' is a loss.
        We return the last memory state which can be then used as
        a seed if we continue the subsequent training, following
        the considered here T-patch.
        '''       
        Mem={}
        Y={}
        P={}
        Mem[-1]=np.copy(m_seed) 
        L=0
        '''
        Forward pass: 
            1. Calculate the memory vector iteration.            
            2. Calculate the output vector.            
            3. Calculate the probability weights for the output vector.           
            4. Calculate the loss function.
        '''
        for t in xrange(self.T):
            Mem[t]=np.tanh(np.dot(self.W,x[t])+np.dot(self.V,Mem[t-1])+self.b)  
            Y[t]=np.dot(self.U,Mem[t])+self.e
            P[t]=np.exp(Y[t])/np.sum(np.exp(Y[t]))  
            L+=-np.log(P[t][y[t],0])          
        '''
        Backward pass:            
            Iterate from the higher to lower times/layers. Calculate the
            gradients of the loss function w.r.t. parameters of the RNN.
            Initialize those gradients to zero arrays of the same shape
            as the corresponding parameters.     
            For each t we will need gradient of the loss function w.r.t.
            memory vector at t, as is obtained through the dependence of
            the loss function on the memory vector at t+1. Save it
            as next_nabla_m at given t and use it when calculation gradients
            at t-1. At t=T there's no next time step, so we initialize
            next_nabla_m=0. We also calculate auxiliary nabla_m_hat, which
            is a derivative of the loss function w.r.t. the argument of
            the tanh() activation function.
        '''
        nabla_W=np.zeros((self.M,self.S))
        nabla_V=np.zeros((self.M,self.M))
        nabla_U=np.zeros((self.N,self.M))
        nabla_b=np.zeros((self.M,1))
        nabla_e=np.zeros((self.N,1))
        next_nabla_m=np.zeros((self.M,1))
        for t in reversed(xrange(self.T)):  
            nabla_y=np.copy(P[t])
            nabla_y[y[t]]-=1 
            nabla_U+=np.dot(nabla_y,Mem[t].T)
            nabla_e+=nabla_y
            nabla_m=np.dot(self.U.T,nabla_y)+next_nabla_m 
            nabla_m_hat=(1-Mem[t]*Mem[t])*nabla_m
            nabla_b+=nabla_m_hat
            nabla_W+=np.dot(nabla_m_hat,x[t].T)  
            nabla_V+=np.dot(nabla_m_hat,Mem[t-1].T) 
            next_nabla_m=np.dot(self.V.T,nabla_m_hat)
        '''
        Clip the gradients to fix exploding cases.
        '''      
        np.clip(nabla_W,-5,5,out=nabla_W)
        np.clip(nabla_V,-5,5,out=nabla_V)
        np.clip(nabla_U,-5,5,out=nabla_U)
        np.clip(nabla_b,-5,5,out=nabla_b)
        np.clip(nabla_e,-5,5,out=nabla_e)
        return L,nabla_W,nabla_V,nabla_U,nabla_b,nabla_e,Mem[self.T-1]
    def train(self,p,m_seed):
        '''        
        Arguments: 
            p -- start index in the T-sequence. 
            m_seed -- pre-start memory vector seed.
        Returns:   
            p,m_seed -- (the new start index of the sequence,
                        the new seed memory state looped from the last memory
                        state calculated in the forward pass).
        1. Check whether new T-sequence will spill over the end of the data,
           if yes, reset the seed memory vector and set the beginning of
           the T-sequence to the start of the data.
        2. Prepare arguments x, y, m_seed for the gradient function.       
        3. Calculate the loss,gradients, and the (T-1) last memory state.
        4. Set the new m_seed to the (T-1) last memory state, creating 
           a loop for the memory training, continuous across the data self.A.
        5. Update the loss Loss.
        6. Shift the parameters in the direction of negative gradients and
        update norms_W... arrays with the new gradient norms.
        ''' 
        if p+self.T>=len(self.A): 
            m_seed=np.zeros((self.M,1)) 
            p=0       
        x=self.A[p:p+self.T]
        y=self.B[p:p+self.T]
        L,nabla_W,nabla_V,nabla_U,nabla_b,nabla_e,m_seed=\
                                            self.gradients(x,y,m_seed)
        self.Loss=0.999*self.Loss+0.001*L
        for par,nabla_par,norm in\
        zip([self.W,self.V,self.U,self.b,self.e],\
            [nabla_W,nabla_V,nabla_U,nabla_b,nabla_e],\
            [self.norms_W,self.norms_V,self.norms_U,self.norms_b,self.norms_e]):
            norm+=nabla_par*nabla_par
            par+=-self.r*nabla_par/np.sqrt(norm+1e-8) 
        p+=self.T   
        return p,m_seed       
    def sample(self,x_start,l,m_start):
        '''  
        ***  To be applied when S=N  ***
        Arguments:
            x_start -- start input vector for the sample. 
            l -- length of the sample.
            m -- seed memory,.           
        Returns:
            list of length n of generated output prediction numbers.
            Each of these numbers is a choice of possible output states
            picked with a probability determined by the N-vector weight
            output of the given t-layer.
        Sample a sequence of size n, starting with the element x_start.
        Since passed vectors are mutable, copy them. Each predicted 
        element will be fed as an input for the next prediction, that's
        why we need S=N.
        '''
        x=np.copy(x_start)
        m=np.copy(m_start)
        predictions=[]
        for t in xrange(l):
            m=np.tanh(np.dot(self.W,x)+np.dot(self.V,m)+self.b)
            y=np.dot(self.U,m)+self.e
            prob=np.exp(y)/np.sum(np.exp(y))
            pred=np.random.choice(range(self.N),p=prob.ravel())
            x=np.zeros((self.S,1))
            x[pred]=1
            predictions+=[pred]      
        return predictions 
    def run_samples(self,map_to_char):
        '''
        ***  To be applied when S=N  ***
        Runs the tradining iterations  until interrupted.
        The iteration number is n, the training is done on batches
        We need S=N so that we will be able to sample().
        If sample() is omitted we don't need to impose S=N.
        '''
        n=0
        p=0
        m_seed=np.zeros((self.M,1)) 
        while True:
            if n%100 == 0:
                print 'iter %d, loss: %f'%(n,self.Loss)
                x=self.A[p]
                sample=self.sample(x,200,m_seed)
                txt=''.join(map_to_char[i] for i in sample)
                print '----\n %s \n----' % (txt, )
            p,m_seed=self.train(p,m_seed)
            n+=1
            
'''
*******************************************************************************
Read data from simple plaint text file 'input.txt'.
Will have S=N, that is, the same size of the imput and output states.
Then we can run_samples() and take samples of the text while training the RNN. 
*******************************************************************************
'''

data=open('input.txt', 'r').read()
chars=list(set(data)) # alphabet
T=25 # unroll the NN into T steps
S=len(chars) # each character will be represented as 1-in-S vector of size S
N=S
M=200 #size of hidden layer of neurons
char_to_ix={ch:i for i,ch in enumerate(chars)}
ix_to_char={i:ch for i,ch in enumerate(chars)}
A=[]
B=[]
r=0.1
for c in data:
    a=np.zeros((S,1))
    ix=char_to_ix[c]
    a[ix]=1
    A+=[a]
    B+=[ix]
B=B[1:]+[B[0]]
rnn=RNN(T,S,M,N,A,B,r)
rnn.run_samples(ix_to_char)
