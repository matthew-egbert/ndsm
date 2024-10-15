# type: ignore

    #### PHASE PLOT
    def phase_plot() : ## from pattern experiment
        figure(figsize=(8,8))
        theta = time%(training_pattern_length*DT) / (training_pattern_length*DT) * 2*np.pi
        r = sms[:,0] * time
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        plot(theta, r,'.',ms=1,label='sensor')
        title('$r=t; θ=sensor$')
        #ax.set_rmax(2)
        gca().set_rticks([])  # Less radial ticks
        gca().set_rlabel_position(-2.5)  # Move radial labels away from plotted line
        #legend()
        #gca().grid(False)
        tight_layout()
        savefig(path+'phase.png',dpi=300)

    def cleanplot() :## from pattern experiment -- not sure if was useufl or not
        xticks([])
        plt.box(False)
        xlim(0,time[-1])

    def timeseries_plot() : ## from pattern experiment -- not sure if was useufl or not
        #### TIMESERIES PLOTS
        figure(figsize=(8,5))

        r,c = 4,1
        subplot2grid((r,c),(0,0))
        fill_between(time,0*sms[:,0]-1,sms[:,-2],label='lm',step='pre')
        plot(time[:-training_pattern_length],running_average(sms[:,-2],training_pattern_length)[:-training_pattern_length],lw=0.7,color='k',label='running average')
        cleanplot()
        ylabel('LM')

        subplot2grid((r,c),(1,0))
        fill_between(time,0*sms[:,0]-1,sms[:,-1],label='rm',step='pre',facecolor='g')
        plot(time[:-training_pattern_length],running_average(sms[:,-1],training_pattern_length)[:-training_pattern_length],lw=0.7,color='k',label='running average')
        cleanplot()
        ylabel('RM')

        # subplot2grid((r,c),(2,0))
        # fill_between(time,0*sms[:,0],sms[:,0],label='sensor',step='pre',facecolor='y')
        # plot(time[:-training_pattern_length],running_average(sms[:,0],training_pattern_length)[:-training_pattern_length],lw=0.7,color='k',label='running average')
        # cleanplot()
        # ylabel('sensor')
        # ticks = arange(0,time[-1],training_pattern_length*DT)
        # xticks(ticks)
        # gca().set_xticklabels([f' ' for t in ticks])
        # gca().xaxis.set_ticks_position('top')

def phase_plot_range(α=0,ω=1024*4,use_position=True,color='k') : ## was for osc experiment, not sure if it is useful or not
        for i in range(α,ω-2):
            if use_position :
                xx = x[i:i+2]  
            else :
                """otherwise use sensor state"""
                xx = copy(sms[i:i+2,0])
                xx[0] += 0.00005*(i)
                xx[1] += 0.00005*(i+1)
            plot(xx,
                 sms[i:i+2,1],alpha=0.2,lw=0.5,color=color)
        yticks(list(set(sms[:,1])))
        if use_position :
            xlim(-2.5,2.5)
            xlabel('x')
            title(f'$t '+'\\in'+f' [{time[α]:.2f},{time[ω]:.2f}]$')
        else :
            xlim(-0.05,1)
            xlabel('s')
            xticks(list(set(sms[:,0])))
        ylabel('m',rotation=0)
        

    def phase_plot() : ## was for osc experiment, not sure if it is useful or not
        figure(figsize=(6.0,5.0))            

        subplot2grid((2,2),(0,0))
        phase_plot_range(α=0,ω=1024*4)        
        subplot2grid((2,2),(0,1))
        phase_plot_range(α=-1024*4-2,ω=-2)
        ylabel('')

        subplot2grid((2,2),(1,0))
        phase_plot_range(α=0,ω=1024*4,use_position=False)
        subplot2grid((2,2),(1,1))
        phase_plot_range(α=-1024*4-2,ω=-2,use_position=False)
        ylabel('')

        tight_layout()
        savefig(path+'osc_phase_comparison.png',dpi=300)


    def timeseries_plot() : ## was for osc experiment, not sure if it is useful or not
        #### TIMESERIES PLOT
        figure(figsize=(22,12))
        def cleanplot() :
            xticks([])
            #yticks([])
            plt.box(False)
            # xlim(-1,1)
            # ylim(0,1)
        
        r,c = 5,1
        subplot2grid((r,c),(0,0))
        plot(time,x,lw=0.7,color='k')
        cleanplot()
        ylabel('x')
        ticks = arange(0,time[-1],Ω*DT)
        xticks(ticks)
        gca().set_xticklabels([f' ' for t in ticks])
        gca().xaxis.set_ticks_position('top')
        fill_between([0,training_stop_iteration*DT],[-5,-5],[5,5],alpha=0.2,facecolor='k')

        subplot2grid((r,c),(1,0))
        fill_between(time,0*sms[:,0],sms[:,0],label='sensor',step='pre',facecolor='y')
        plot(time[:-Ω],running_average(sms[:,0],Ω)[:-Ω],lw=0.7,color='k',label='running average')
        cleanplot()
        ylabel('sensor')
        
        subplot2grid((r,c),(2,0))
        fill_between(time,0*sms[:,1]-1,sms[:,1],label='lm',step='pre',alpha=0.5)
        plot(time[:-Ω],running_average(sms[:,1],Ω)[:-Ω],lw=0.7,color='k',label='running average')
        cleanplot()
        ylim(sms[:,1].min(),sms[:,1].max())
        ylabel('M')
            
        subplot2grid((r,c),(3,0),rowspan=2)
        log_prediction_error = np.log(prediction_error,where=prediction_error!=0)
        fill_between(time,log_prediction_error*0-10,log_prediction_error,step='pre',label='log($\\epsilon$)',facecolor=red,alpha=0.7)
        plot(time[:-Ω],running_average(log_prediction_error,Ω)[:-Ω],lw=0.7,color='k',label='running average')
        ylim(-4,1.8)
        ylabel('log($\\epsilon$)')
        cleanplot()
        ticks = arange(0,time[-1],Ω*DT)
        xticks(ticks)
        ticklabels = [f' ' for t in ticks]
        ticklabels[0] = '0'
        ticklabels[-1] = f'{time[-1]:.0f}'
        ticklabels[len(ticklabels)//2] = f'{time[-1]/2:.0f}'
        ticklabels[len(ticklabels)//4] = f'{time[-1]/4:.0f}'
        ticklabels[3*len(ticklabels)//4] = f'{3*time[-1]/4:.0f}'
        xlabel('time')
        gca().set_xticklabels(ticklabels)
        tight_layout()
        savefig(path+'timeseries.png',dpi=300)
        show()