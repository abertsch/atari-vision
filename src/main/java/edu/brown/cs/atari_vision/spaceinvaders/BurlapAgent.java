package edu.brown.cs.atari_vision.spaceinvaders;

import burlap.behavior.policy.Policy;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.shell.visual.VisualExplorer;
import burlap.visualizer.Visualizer;
import edu.brown.cs.atari_vision.ale.burlap.ALEEnvironment;
import edu.brown.cs.atari_vision.ale.burlap.ALEStateGenerator;

/**
 * Created by MelRod on 3/22/16.
 */
public class BurlapAgent {

    Policy policy;
    SADomain domain;
    ALEEnvironment env;
    Visualizer vis;

    public BurlapAgent(Policy policy, SADomain domain, ALEStateGenerator stateGenerator, Visualizer vis, String rom, boolean useGUI) {

        this.domain = domain;
        this.policy = policy;
        this.vis = vis;

        env = new ALEEnvironment(domain, stateGenerator, rom, useGUI);
    }

    public void run(int episodes) {
        if (vis != null) {
            VisualExplorer exp = new VisualExplorer(domain, env, vis);
            exp.initGUI();
            exp.startLiveStatePolling(10);
        }

        for (int e = 1 ; e <= episodes ; e++) {
            runEpisode();

            env.resetEnvironment();
        }

        // Clean up the environment
        env.die();

    }

    public void runEpisode() {
        while(!env.isInTerminalState()){
            State curState = env.currentObservation();
            Action action = policy.action(curState);

            env.executeAction(action);
        }
    }
}
