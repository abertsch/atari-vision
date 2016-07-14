package edu.brown.cs.atari_vision.spaceinvaders;

import burlap.behavior.policy.Policy;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.shell.visual.VisualExplorer;
import burlap.visualizer.Visualizer;
import edu.brown.cs.burlap.ALEEnvironment;

/**
 * Created by MelRod on 3/22/16.
 */
public class BurlapAgent {

    static final String alePath = "/home/maroderi/projects/Arcade-Learning-Environment/ale";

    Policy policy;
    SADomain domain;
    ALEEnvironment env;
    Visualizer vis;

    public BurlapAgent(Policy policy, SADomain domain, Visualizer vis, String rom) {

        this.domain = domain;
        this.policy = policy;
        this.vis = vis;

        env = new ALEEnvironment(alePath, rom);
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

    }

    public void runEpisode() {
        while(!env.isInTerminalState()){
            State curState = env.currentObservation();
            Action action = policy.action(curState);

            env.executeAction(action);
        }
    }
}
