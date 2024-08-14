       plms = np.array(())
        for h,plm,prm in zip(self.all_possible_histories,plms,plms) :
            #print(f'{list(flatten(h))} : {plm:.3f}, {prm:.3f}')
            probs[str(list(flatten(h)))] = (plm,prm)    

        step(range(len(self.all_possible_histories)),plms,where='mid',label='p(lm|sensor history)')
        step(range(len(self.all_possible_histories)),prms,where='mid',label='p(rm|sensor history)')
        gca().set_xticks(range(len(self.all_possible_histories)))
        gca().set_xticklabels([str(list(h)) for h in self.all_possible_histories],rotation=90)
        title('Likelihood of motor values for sensor history') 
        legend()
        tight_layout()
        savefig('p(m|s_h).png')


   def onehot_to_motor_states(self, onehot) :
        combined_index = argmax(onehot)
        lmi = combined_index // self.N_ALLOWED_MOTOR_VALUES
        rmi = combined_index % self.N_ALLOWED_MOTOR_VALUES
        lm = self.ALLOWED_MOTOR_VALUES[lmi]
        rm = self.ALLOWED_MOTOR_VALUES[rmi]
        return lm,rm

    def motor_states_to_onehot(self, motor_states) :                        
        lm,rm = motor_states
        lmi = np.digitize(lm,self.ALLOWED_MOTOR_VALUES,right=True)
        rmi = np.digitize(rm,self.ALLOWED_MOTOR_VALUES,right=True)
        combined_index = lmi*self.N_ALLOWED_MOTOR_VALUES + rmi
        one_hot = zeros(self.N_ALLOWED_MOTOR_VALUES**2)
        one_hot[combined_index] = 1.0
        #print(f'lm: {lm} -> lmi: {lmi} || rm: {rm} -> rmi: {rmi} || combined_index: {combined_index} \none_hot: {one_hot}')
        return one_hot