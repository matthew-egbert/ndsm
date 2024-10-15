import pickle
from pylab import *
import networkx as nx

from discval import DiscVal, OneHotter

def network_plot(time,sms,α=0,ω=-1,path='',filename='network.png',label='') :
    if α < 0 :
        α = len(sms)+α
    if ω < 0 :
        ω = len(sms)+ω
    allowed_motor_values = np.linspace(-0.3,0.3,5)
    allowed_sensor_values = np.linspace(0,1,11)
    
    os = DiscVal(allowed_sensor_values, 0, name = "OS")
    om = DiscVal(allowed_motor_values, 0, name = "OM")
    oh = OneHotter([os,om])
    
    ## num sensorimotor states
    N = len(oh.onehot)
    
    transition_counts = np.zeros((N,N))
    
    label_mapping = dict()
    pos = dict()
    
    for i in range(α,ω-2):
        s1,m1 = sms[i]
        s2,m2 = sms[i+1]
        oh.values = [s1,m1]
        i1 = np.argmax(oh.onehot)
        label_mapping[i1] = f'{s1:.2f},{m1:.2f}' # f'{s1*10:.0f},{m1*10:.0f}'
        pos[label_mapping[i1]] = (m1,s1)
        

        oh.values = [s2,m2]
        i2 = np.argmax(oh.onehot)

        transition_counts[i2,i1] += 1
    
    transition_counts = log(transition_counts+1)
    figure(figsize=(6,4.5))
    imshow(transition_counts,cmap='viridis')
    colorbar()
    tight_layout()
    savefig(path+'transition_count_matrix.png',dpi=300)
    close()
    
    figure(figsize=(4.5,4.5))
    xlim(-0.35,0.35)
    ylim(-0.05,0.98)
    x0, y0 = gca().transAxes.transform((-0.35, -0.05)) # lower left in pixels
    x1, y1 = gca().transAxes.transform((0.35, 0.98)) # upper right in pixes
    dy = x1 - x0
    dx = y1 - y0
    maxd = max(dx, dy)
    width = .02 * maxd / dx
    height = .02 * maxd / dy
    
    G = nx.DiGraph(transition_counts)
    G.remove_nodes_from(list(nx.isolates(G)))        
    G = nx.relabel_nodes(G, label_mapping)        
    # for node in G.nodes() :
    #     G.nodes[node]['pos'] = (float(node.split(',')[0]), float(node.split(',')[1]))
    
    #nx.draw(G, pos, with_labels=True, node_size=200, node_color='skyblue', font_size=12, font_color='black', font_weight='bold', edge_color='gray', width=0.5, arrowsize=5)
    #for node in G.nodes() :
        #gca().add_artist(patches.Ellipse((pos[node][0], pos[node][1]), width, height))
        #text = gca().text(pos[node][0],pos[node][1],node,ha='center',va='center',fontsize=10)
        #gca().add_artist(text)
    for edge in G.edges():
        source, target = edge
        rad = 0.2
        arrowprops=dict(lw=G.edges[(source,target)]['weight'],
                        arrowstyle="->",
                        color='black',
                        connectionstyle=f"arc3,rad={rad}",
                        linestyle= '-',
                        shrinkA=2.0,
                        shrinkB=2.0,
                        alpha=1.0)
        gca().annotate("",
                    xy=pos[source],
                    xytext=pos[target],
                    arrowprops=arrowprops
                )
    
    # xticks([])
    # yticks([])
    xlabel('m',fontsize=12)
    ylabel('s',fontsize=12,rotation=0)
    plt.text(0.0, 1.0, label, fontsize=28, ha='center', fontweight='bold', transform=plt.gca().transAxes)
    plt.text(0.5,1.0,f'$t \\in[{time[α]:.2f},{time[ω]:.2f})$',fontsize=14,ha='center', transform=plt.gca().transAxes)
    plt.box(False)
    tight_layout()
    savefig(path+filename,dpi=300)

def oscillation_network_plots(path) :
    time = np.load(path+'time.npy')
    sms = np.load(path+'sms.npy')
    po = pickle.load(open(path+'pickle_objs.pkl','rb'))
    DT = po['DT']

    p = int(3.11//DT) ## oscilalation length in iterations (approximated visually)
    n = 36 ## number of oscillations
    print(sms.shape)
    network_plot(time,sms,0,p*n,path=path,filename='network_start.png',label='A')
    network_plot(time,sms,-p*n,-1,path=path,filename='network_end.png',label='B')