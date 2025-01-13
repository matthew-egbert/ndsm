from pylab import *
from scipy.stats import entropy
#from sklearn.feature_selection import mutual_info_classif

def trajectory(n_its) :
    """ Compute a trajectory of Polya's urn with n_its iterations """
    state = [1,1]
    state_h = []

    def sample(n0,n1) :
        """ Sample a ball from the urn """
        return np.random.choice([0,1],p=[n0/(n0+n1),n1/(n0+n1)])
    
    for it in range(n_its) :
        state_h.append([state[0],state[1]])
        ball = sample(state[0],state[1])
        state[ball] += 1
        
    return state_h

def plot_trajectory(traj) :
    plot(traj[:,0]/(traj[:,0]+traj[:,1]),alpha=0.7)
    #xscale('log')
    ylim(0,1)
    xlabel('Iteration')
    ylabel('Fraction of black marbles')

N_ITS = 500
N_TRAJ = 1

cols = []
targets = []
for i in range(N_TRAJ) :    
    traj = np.array(list(trajectory(N_ITS)))
    b = traj[:,0]
    
    flips = np.diff(b)
    cols.append(flips)    

    target = traj[-1,0]/(traj[-1,0]+traj[-1,1])
    targets.append(target)
    plot_trajectory(traj)
savefig('trajectory1.png',dpi=300)

# cols = np.array(cols)
# targets = np.array(targets).reshape(-1,1)
# final_state_bins = linspace(0,1,11)
# targets_binned = final_state_bins[np.digitize(targets,final_state_bins)]

# target_entropy = entropy(histogram(targets_binned,final_state_bins)[0],base=2)

# col_entropies = []
# joint_entropies = []
# mis = []

# for col_i in range(cols.shape[1]) :
#     one_count = sum(cols[:,col_i])
#     zero_count = np.shape(cols)[0] - one_count
#     col_entropy = entropy([zero_count,one_count],base=2)
#     col_entropies.append(col_entropy)

#     joint = [flip + final for flip,final in zip(cols[:,col_i],targets_binned)]
#     joint_hist = histogram(joint,linspace(0,2,11))[0]
#     joint_entropy = entropy(joint_hist,base=2)
#     joint_entropies.append(joint_entropy)

#     mi = col_entropy + target_entropy - joint_entropy
#     print(f'MI: {mi} = {col_entropy} + {target_entropy} - {joint_entropy}')
#     mis.append(mi)

# figure()
# plot(mis)
# xlabel('Iteration')
# ylabel('I(selection;final ratio)')
# #xscale('log')
# savefig('mutual_information.png',dpi=300)
# show()
