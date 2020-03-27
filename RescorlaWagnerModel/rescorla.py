Our Rescorla Wagner Model
#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt


# In[4]:


#Question 1a

def v(t, a, b, val):
    lst = []
    while t <= 20:
        if len(lst) < 22:
            delta_v = a * b * (1 - val)
            val = val + delta_v
            lst.append(val)
            t += 1
    return lst


# In[5]:


x = np.arange(1, 22)
z = v(0, .5, .1, .05)
y = np.asarray(z)


# In[6]:


plt.figure()
plt.title('Question 1a')
plt.xlabel('time', fontsize = 10)
plt.ylabel('learning', fontsize = 10)
plt.plot(x, y)
plt.show()


# In[7]:


k = v(0, .5, .1, .5)
p = np.arange(1, 22)
r = np.asarray(k)

plt.figure()
plt.title('Question 1a')
plt.xlabel('time', fontsize = 10)
plt.ylabel('learning', fontsize = 10)
plt.plot(p, r)
plt.show()


# In[8]:


type(r)


# In[9]:


type(p)


# In[10]:


#Question 1b
#By increasing permitted length of list, I see .8 is reached at trial 30

def trials(t, a, b, val):
    mylst = []
    while t <= 30:
        if len(mylst) < 100:
            delta_v = a * b * (1 - val)
            val = val + delta_v
            mylst.append(val)
            t += 1
    return mylst


# In[11]:


howmanytrials = trials(0, .5, .1, .05)


# In[12]:


howmanytrials[30]


# In[207]:


#Question 1b
'''It takes 30 trials to reach Vlight = 0.8 if association is 0.05'''


# In[14]:


#Question 1c

thirteentrials = trials(0, 1.2, .1, 0)


# In[209]:



'''we see that association exceeds 0.8 on trial 13 when salience value is = 1.2'''


# In[17]:


#Question 2

def vlight(x):
    a = 0.5
    b = 0.1
    if x == 0:
        return .8
    else:
        return vlight(x-1) + a * b * (1 - vlight(x-1))



def vbell(x):
    a = 0.2
    b = 0.1
    if x == 0:
        return 0
    else:
        return vbell(x-1) + a * b * (1 - vlight(x-1) - vbell(x-1))


# In[18]:


#Question 2
mini = []
for i in range(21):
    mini.append(vbell(i))
mini = np.asarray(mini)

plt.figure()
plt.title('Question 2')
plt.xlabel('trials', fontsize = 10)
plt.ylabel('association strength', fontsize = 10)
plt.plot(mini)
plt.show()


# In[158]:


#Question 3a
def bellwfood(x):
    a = 0.2
    b = 0.1
    if x == 0:
        return 0
    elif x % 2 == 0:
        return bellwfood(x-1) + a * b * (1 - bellwfood(x-1))
    else:
        return bellwfood(x-1) - a * b * (bellwfood(x-1))


# In[159]:


empt = []

for i in range(20):
    empt.append(bellwfood(i))

empt = np.asarray(empt)


# In[160]:


empt


# In[211]:


#Question 3a
plt.figure()
plt.title('Question 3a')
plt.xlabel('trials', fontsize = 10)
plt.ylabel('association strength', fontsize = 10)
plt.plot(empt)
plt.show()

'''Here we see in the data that the first trial with food and bell creates an association. The next trial (extinction trial) slightly lowers the association between bell and food as there is no food rewarded. This pattern of increasing association then slight decrease after extinction continues and alternates up to an eventual association strength of ~ .15 after 20 trials.'''


# In[199]:


#Question 3b
a = 0.2
b = 0.1
def random_trials(x, P):
    random = np.random.randint(1, 100)
    probability = np.arange(1, P+1)
    if x == 0 or P == 0:
        return 0.1
    if random < P:
        return random_trials((x-1), P) + a * b * (1 - random_trials((x-1), P))
    elif random >= P:
        return random_trials((x-1), P) - a * b * (random_trials((x-1), P))
    else:
        return None


# In[187]:


random_trials(10, 40) #this is just a test line of code to ensure I'm getting an output


# In[200]:


#below I make four separate trials, trial25, trial50, trial75, and trial90
#I then plot each of these trials. It can be seen that as P increases, the association becomes
#stronger and more consistent

trial_25 = []
for i in range(20):
    trial_25.append(random_trials(i, 25))


# In[201]:


trial_50 = []
for i in range(20):
    trial_50.append(random_trials(i, 50))


# In[202]:


trial_75 = []
for i in range(20):
    trial_75.append(random_trials(i, 75))


# In[ ]:


trial_90 = []
for i in range(20):
    trial_90.append(random_trials(i, 90))


# In[203]:


plt.figure()
plt.title('P = 25')
plt.xlabel('trials', fontsize = 10)
plt.ylabel('association strength', fontsize = 10)
plt.plot(trial_25)
plt.show()


# In[204]:


plt.figure()
plt.title('P = 50')
plt.xlabel('trials', fontsize = 10)
plt.ylabel('association strength', fontsize = 10)
plt.plot(trial_50)
plt.show()


# In[205]:


plt.figure()
plt.title('P = 75')
plt.xlabel('trials', fontsize = 10)
plt.ylabel('association strength', fontsize = 10)
plt.plot(trial_75)
plt.show()


# In[213]:


plt.figure()
plt.title('P = 90')
plt.xlabel('trials', fontsize = 10)
plt.ylabel('association strength', fontsize = 10)
plt.plot(trial_90)
plt.show()


# In[222]:


#Question 3b
''' Marrs computational level asks: what problem does this system solve? Is the organism optimal in some sense? Well, this system solves the problem of trying to understand how asssociation strength changes when the probability of a traditional reward-learning task changes. This helps to model how reward learning takes places and whether strong associations will be maintained even when there is a high probability of extinction. As shown in the graphs, when extinction is more likely, a strong association is less likely. The organism, in my opinion, is optimal in the sense that it creates the desired computation with relatively few lines of code. The problem that we chose to solve (to model association strength of conditioned and unconditioned stimulus) has effectively been solved.'''


# In[223]:


#Question 4
'''Salience is the prominence or significance of the stimulus (e.g. a small and barely perceptible tone vs a loud horn) while learning rate is simply the rate at which the learned association takes place. Despite the fact that there are clear psychological nuances and differences between these two things, the difference is hard to code computationally. An experiment to disentagle the two might ask a subject to press a big red button with their left hand as soon as they experience a shock on their right hand. Afterwards, the subject will also rate the intensity (salience) of the shock they receive on their right hand. This may show that learning rate is consistent while salience of the stimuli changes significantly each trial, thus disentangling the two. '''


# In[ ]:
