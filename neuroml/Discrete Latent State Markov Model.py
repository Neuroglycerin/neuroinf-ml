
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')


# In[2]:

creaks = np.array([[0.1, 0.1, 0.1, 0.9, 0.9],
                   [0.1, 0.9, 0.9, 0.1, 0.1],
                   [0.1, 0.9, 0.1, 0.1, 0.1],
                   [0.9, 0.1, 0.1, 0.9, 0.9],
                   [0.9, 0.9, 0.1, 0.1, 0.1]])


# In[3]:

bumps = np.array([[0.9, 0.1, 0.1, 0.9, 0.9],
                  [0.9, 0.1, 0.1, 0.9, 0.9],
                  [0.1, 0.9, 0.1, 0.9, 0.1],
                  [0.1, 0.9, 0.1, 0.1, 0.1],
                  [0.9, 0.1, 0.1, 0.1, 0.1]])


# In[4]:

observations = np.array([[1, 0],
                         [0, 1],
                         [1, 1],
                         [0, 0],
                         [1, 1],
                         [0, 0],
                         [0, 1],
                         [0, 1],
                         [0, 0],
                         [0, 0]])


# Don't want to write out the full matrix of transition probabilities, because it'd be way too big.
# Would much rather just write a function that will return the transition probabilities based on 
# the current location.
# 
# Going to take two inputs that are each locations (as tuples) and check if they're adjacent.
# If they are, then the transition probability is calculated as uniform over the possible options.

# In[5]:

def transition_probability(state_from, state_to, N=5):
    """
    Given a coordinate, return 
    the transition probabilities.
    """
    
    prob = 0.0
    # Check if adjacent
    if abs(state_from[0] - state_to[0]) + abs(state_from[1] - state_to[1]) == 1:
        # Check for x edge
        x_edge = False
        y_edge = False
        if state_from[0] == 0 or state_from[0] == N-1:
            x_edge = True
        #Check for y edge
        if state_from[1] == 0 or state_from[1] == N-1:
            y_edge = True

        if x_edge and y_edge:
            prob = 1.0 / 2
        elif x_edge or y_edge:
            prob = 1.0 / 3
        else:
            prob = 1.0 / 4
    
    return prob


# In[6]:

N=5
A = np.zeros([25,25])

for i in range(0,N**2):
    for j in range(0,N**2):
        A[i,j] = transition_probability([i %N, i / N], [j % N, j / N])


# Filtering
# =========
# 
# Now we're going to actually start doing some HMM.
# First, we're going to flatten the emission probabilities into vectors to make it 
# consistent with the textbook. Then, putting it together into a full emission matrix $B$.
# 
# 
# 

# In[7]:

creaks = np.ndarray.flatten(creaks)
bumps = np.ndarray.flatten(bumps)


# In[8]:

# emission probabilities
B = np.vstack([creaks,bumps])


# Look at Barber p.496:
# 
# $$
# p(h_{t},v_{1:t}) = \sum_{h_{t-1}} p(v_{t}|h_{t}) p(h_{t}|h_{t-1}) p(h_{t-1},v_{1:t-1})
# $$
# 
# Hence, we define:
# 
# $$
# \alpha(h_{t}) = p(h_{t}, v_{1:t})
# $$
# 
# This defines the $\alpha$-recursion:
# 
# $$
# \alpha(h_{t}) = p(v_{t}|h_{t}) \sum_{h_{t-1}} p(h_{t}| h_{t-1}) \alpha(h_{t-1})
# $$
# 

# In[9]:

if observations[0][1] == 1:
    b = bumps
else:
    b = 1-bumps
if observations[0][0] == 1:
    c = creaks
else:
    c = 1-creaks


# In[130]:

ht_dist = np.ones([N*N]) * 1.0 / (N * N)
alpha = b*c*ht_dist


# In[131]:

alpha = alpha[np.newaxis].T


# In[124]:

def alpha_update(alpha, t):
    """Defined as joint distribution of h_t and v_1:t.
    Return a discrete probability distribution."""
    
    if observations[t][1] == 1:
        b = bumps
    else:
        b = 1-bumps
    if observations[t][0] == 1:
        c = creaks
    else:
        c = 1-creaks
    x = np.dot(A,alpha)
    alpha = b[np.newaxis].T*c[np.newaxis].T * np.dot(A,alpha)
    return alpha


# In[132]:

t = 1


# In[133]:

plot_heatmap(alpha/sum(alpha))


# In[134]:

alpha = alpha_update(alpha, t)
ht_dist = alpha/sum(alpha)
t = t+1
print(t)
plot_heatmap(ht_dist)


# In[102]:

def plot_heatmap(v):
    """
    Plot a heatmap given a vector of 
    state probabilities.
    """
    N = int(np.sqrt(len(v)))
    imshow(v.reshape([N,N]),cmap = cm.Greys_r)
    return None


# In[100]:

ht_dist.reshape([5,5])


# In[99]:

imshow(np.random.randn(5,5))


# In[37]:

print(ht_dist.shape)


# In[67]:

x = np.random.randn(3,1)


# In[68]:

y = np.random.randn(3,1)


# In[69]:

x


# In[70]:

y


# In[71]:

x*y

