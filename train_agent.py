#!/usr/bin/env python

import torch
import pickle
import numpy as np
import time
import os
from shutil import copyfile

from model import RNN
from data_structs import Vocabulary, Experience
from scoring_functions import get_scoring_function
from utils import Variable, seq_to_smiles, fraction_valid_smiles, unique, sa_score, percentage_easy_sa, percentage_unique, check_and_save_smiles
from vizard_logger import VizardLog
from scipy.stats import levy

def train_agent(restore_prior_from='data/Prior.ckpt',
                restore_agent_from='data/Prior.ckpt',
                scoring_function='tanimoto',
                scoring_function_kwargs=None,
                save_dir=None, learning_rate=0.0005,
                batch_size=16, lb=2, ub=3, n_steps=3000,
                num_processes=0, sigma=20, sigma_mode='static',
                lambda_1=2, lambda_2=2, experience_replay=0):
    print('started training')
    print('sigma used: ' + str(sigma))
    print('sigma mode: ' + sigma_mode)
    voc = Vocabulary(init_from_file="data/Voc")

    start_time = time.time()

    Prior = RNN(voc)
    Agent = RNN(voc)

    logger = VizardLog('data/logs')

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    if torch.cuda.is_available():
        Prior.rnn.load_state_dict(torch.load('data/Prior.ckpt'))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
    else:
        Prior.rnn.load_state_dict(torch.load('data/Prior.ckpt', map_location=lambda storage, loc: storage))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=0.0005)

    if sigma_mode == 'static':
        xtb_run = 'run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()) + '_sf_' + scoring_function + '_sm_' + sigma_mode + '_s_' + str(sigma) + '_bs_' + str(batch_size)+ '_lb_' + str(lb) + '_ub_' + str(ub) + '_ns_' + str(n_steps) + '_l2_' + str(lambda_2)
    else:
        xtb_run = 'run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()) + '_sf_' + scoring_function + '_sm_' + sigma_mode + '_l1_' + str(lambda_1) + '_bs_' + str(batch_size) + '_lb_' + str(lb) + '_ub_' + str(ub) + '_ns_' + str(n_steps) + '_l2_' + str(lambda_2)


    # Scoring_function
    scoring_function_call = get_scoring_function(scoring_function=scoring_function, num_processes=num_processes, batch_size=batch_size, upper_limit=ub, lower_limit=lb, xtb_run=xtb_run,
                                            **scoring_function_kwargs)

    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Experience(voc)

    # Log some network weights that can be dynamically plotted with the Vizard bokeh app
    logger.log(Agent.rnn.gru_2.weight_ih.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_ih")
    logger.log(Agent.rnn.gru_2.weight_hh.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_hh")
    logger.log(Agent.rnn.embedding.weight.cpu().data.numpy()[::30], "init_weight_GRU_embedding")
    logger.log(Agent.rnn.gru_2.bias_ih.cpu().data.numpy(), "init_weight_GRU_layer_2_b_ih")
    logger.log(Agent.rnn.gru_2.bias_hh.cpu().data.numpy(), "init_weight_GRU_layer_2_b_hh")


    with open("data/mols_filtered.smi", 'r') as f:
        smiles_set = set(line.strip() for line in f)

    # Information for the logger
    step_score = [[], []]
    sa = []
    novel = []
    valid = []
    scores = []
    sigmas = []
    standard_score = []
    save_every_n = 25

    print("Model initialized, starting training...")

    for step in range(n_steps):

        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(batch_size)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # Get prior likelihood and score
        prior_likelihood, _ = Prior.likelihood(Variable(seqs))
        smiles = seq_to_smiles(seqs, voc)

        score_bandgap = scoring_function_call(smiles)
        score = score_bandgap[:, 0]
        bandgap = score_bandgap[:, 1]
        
        batch_score = sum(lb < b < ub for b in bandgap)
        
        # Sigma updating scheme
        if sigma_mode == 'linear_decay':
            rate = 0.1
            sigma = 30-rate*step

        if sigma_mode == 'exponential_decay':
            rate = 0.1
            sigma = 30 * (1 - rate)**step

        if sigma_mode == 'levy_flight':
            x = np.random.random(1)[0]
            sigma = int(levy.pdf(x)*100)
        
        if sigma_mode == 'adaptive':
            threshold = 0.6
            # if step > 0:
            #     threshold = np.mean(scores)
            mean_score = np.mean(score)
            if mean_score > threshold:
                sigma = int(sigma*(1-mean_score))
            else:
                sigma = int(sigma*1/(1-mean_score))

        if sigma_mode == 'uncertainty_aware':
            uncertainty = np.std(score)
            sigma = int(sigma * uncertainty*5) # there's obviously a better way to scale sigma with uncertainty, play around with this

        if sigma_mode == 'uncertainty_aware_inverse':
            uncertainty = np.std(score)
            sigma = int(sigma / uncertainty)

        if sigma_mode == 'prior':
            #lambda_1 = 2
            sigma = np.zeros_like(bandgap)

            for i, (b, prior) in enumerate(zip(bandgap, prior_likelihood)):
                if lb < b < ub:
                    sigma[i] = lambda_1 * np.abs(prior)
            sigma = torch.from_numpy(sigma)


        if sigma_mode == 'prior_greedy':
            #lambda_1 = 2
            sigma = np.zeros_like(bandgap)

            for i, (b, prior) in enumerate(zip(bandgap, prior_likelihood)):
                if lb < b < ub:
                    sigma[i] = lambda_1 * np.abs(prior)
            sigma = torch.from_numpy(sigma)

            r = np.random.uniform(0, 1)

            #epsilon = max(0.9 - 0.1 * step, 0.1)
            epsilon = max(0.5 - (0.1 * step), 0.1)

            if r < epsilon:
               	prior_orig = prior_likelihood.clone()
               	prior_likelihood = torch.from_numpy(np.zeros((len(prior_orig),)))
               	sigma = np.zeros_like(bandgap)
                for i, (b, prior) in enumerate(zip(bandgap, prior_likelihood)):
                    if lb < b < ub:
                        sigma[i] = lambda_1 * np.abs(prior)
                sigma = torch.from_numpy(sigma)
        

        def h(bgap):
            if bgap < lb:
                return (lb - bgap) ** 2
            elif bgap > ub:
                return (bgap - ub) ** 2
            else:
                return 0

        p = torch.from_numpy(np.array([h(bg) for bg in bandgap]))
        sigmas.append(sigma)

        print(len([val for val in bandgap if val < ub and val > lb])/len(bandgap))

        # Calculate augmented likelihood
        #augmented_likelihood = prior_likelihood + sigma * Variable(score) + batch_reward
        augmented_likelihood = prior_likelihood + (sigma * Variable(score)) - (lambda_2*p)
        #augmented_likelihood = (1 - sigma) * prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if experience_replay and len(experience)>4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            #exp_augmented_likelihood = (1 - sigma) * exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Then add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, score, prior_likelihood)
        experience.add_experience(new_experience)

        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print("\n       Step {}   Fraction valid SMILES: {:4.1f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
              step, fraction_valid_smiles(smiles) * 100, time_elapsed, time_left))
        print("  Agent    Prior   Target   Score             SMILES")
        #for i in range(10):
        for i in range(len(smiles)):
            print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                       prior_likelihood[i],
                                                                       augmented_likelihood[i],
                                                                       score[i],
                                                                       smiles[i]))
        # Need this for Vizard plotting
        step_score[0].append(step + 1)
        step_score[1].append(np.mean(score))


        # Log some weights
        logger.log(Agent.rnn.gru_2.weight_ih.cpu().data.numpy()[::100], "weight_GRU_layer_2_w_ih")
        logger.log(Agent.rnn.gru_2.weight_hh.cpu().data.numpy()[::100], "weight_GRU_layer_2_w_hh")
        logger.log(Agent.rnn.embedding.weight.cpu().data.numpy()[::30], "weight_GRU_embedding")
        logger.log(Agent.rnn.gru_2.bias_ih.cpu().data.numpy(), "weight_GRU_layer_2_b_ih")
        logger.log(Agent.rnn.gru_2.bias_hh.cpu().data.numpy(), "weight_GRU_layer_2_b_hh")
        logger.log("\n".join([smiles + "\t" + str(round(score, 2)) for smiles, score in zip \
                            (smiles[:12], score[:12])]), "SMILES", dtype="text", overwrite=True)
        logger.log(np.array(step_score), "Scores")

        # Log SMILES diagnostics
        #sa.append(percentage_easy_sa(smiles))
        novel.append(check_and_save_smiles(smiles, smiles_set, present_filepath="present_smiles.smi", absent_filepath="absent_smiles.smi"))
        #novel.append(percentage_unique(smiles))
        valid.append(fraction_valid_smiles(smiles))
        scores.append(np.mean(score))
        standard_score.append(batch_score/batch_size)
        if not save_dir:
            if sigma_mode == 'static': 
                save_dir = 'data/results/run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()) + '_sf_' + scoring_function + '_sm_' + sigma_mode + '_s_' + str(sigma) + '_bs_' + str(batch_size)+ '_lb_' + str(lb) + '_ub_' + str(ub) + '_ns_' + str(n_steps) + '_l2_' + str(lambda_2)
            else:
                save_dir = 'data/results/run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()) + '_sf_' + scoring_function + '_sm_' + sigma_mode + '_l1_' + str(lambda_1) + '_bs_' + str(batch_size) + '_lb_' + str(lb) + '_ub_' + str(ub) + '_ns_' + str(n_steps) + '_l2_' + str(lambda_2)
            #else:
            #    save_dir = 'data/results/run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()) + '_sf_' + scoring_function + '_sm_' + sigma_mode

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if step % save_every_n == save_every_n - 1:
            np.save(os.path.join(save_dir,'training_log_novel.npy'), np.array(novel))
            np.save(os.path.join(save_dir,'training_log_valid.npy'), np.array(valid))
            np.save(os.path.join(save_dir,'training_log_scores.npy'), np.array(scores))
            np.save(os.path.join(save_dir,'training_log_st_scores.npy'), np.array(standard_score))
            #np.save(os.path.join(save_dir,'training_log_sigmas.npy'), np.array(sigmas))

    # If the entire training finishes, we create a new folder where we save this python file
    # as well as some sampled sequences and the contents of the experinence (which are the highest
    # scored sequences seen during training)
    if not save_dir:
        if sigma_mode == 'static': 
            save_dir = 'data/results/run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()) + '_sf_' + scoring_function + '_sm_' + sigma_mode + '_s_' + str(sigma) + '_bs_' + str(batch_size) + '_lb_' + str(lb) + '_ub_' + str(ub) + '_ns_' + str(n_steps) + '_l2_' + str(lambda_2)
        else:
            save_dir = 'data/results/run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()) + '_sf_' + scoring_function + '_sm_' + sigma_mode + '_l1_' + str(lambda_) + '_bs_' + str(batch_size) + '_lb_' + str(lb) + '_ub_' + str(ub) + '_ns_' + str(n_steps) + '_l2_' + str(lambda_2)
        #else:
        #    save_dir = 'data/results/run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()) + '_sf_' + scoring_function + '_sm_' + sigma_mode
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    copyfile('train_agent.py', os.path.join(save_dir, "train_agent.py"))

    experience.print_memory(os.path.join(save_dir, "memory"))
    torch.save(Agent.rnn.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))

    seqs, agent_likelihood, entropy = Agent.sample(16)
    prior_likelihood, _ = Prior.likelihood(Variable(seqs))
    prior_likelihood = prior_likelihood.data.cpu().numpy()
    smiles = seq_to_smiles(seqs, voc)
    score_bg2 = scoring_function_call(smiles)
    score2 = score_bg2[:, 0]
    
    # Log diagnostics 
    #np.save(os.path.join(save_dir,'training_log_sa.npy'), np.array(sa))
    np.save(os.path.join(save_dir,'training_log_novel.npy'), np.array(novel))
    np.save(os.path.join(save_dir,'training_log_valid.npy'), np.array(valid))
    np.save(os.path.join(save_dir,'training_log_scores.npy'), np.array(scores))
    np.save(os.path.join(save_dir,'training_log_st_scores.npy'), np.array(standard_score))
    #np.save(os.path.join(save_dir,'training_log_sigmas.npy'), np.array(sigmas))

    with open(os.path.join(save_dir, "sampled"), 'w') as f:
        f.write("SMILES Score PriorLogP\n")
        for smiles, score2, prior_likelihood in zip(smiles, score2, prior_likelihood):
            f.write("{} {:5.2f} {:6.2f}\n".format(smiles, score2, prior_likelihood))

if __name__ == "__main__":
    train_agent()
