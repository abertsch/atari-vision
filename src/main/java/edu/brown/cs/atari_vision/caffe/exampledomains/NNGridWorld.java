package edu.brown.cs.atari_vision.caffe.exampledomains;

import burlap.behavior.functionapproximation.ParametricFunction;
import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.common.SinglePFTF;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.oo.propositional.PropositionalFunction;
import burlap.mdp.core.oo.state.OOState;
import burlap.mdp.core.oo.state.ObjectInstance;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.common.UniformCostRF;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.shell.visual.VisualExplorer;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import edu.brown.cs.atari_vision.caffe.action.ActionSet;
import edu.brown.cs.atari_vision.caffe.experiencereplay.FixedSizeMemory;
import edu.brown.cs.atari_vision.caffe.learners.DeepQLearner;
import edu.brown.cs.atari_vision.caffe.policies.AnnealedEpsilonGreedy;
import edu.brown.cs.atari_vision.caffe.training.SimpleTrainer;
import edu.brown.cs.atari_vision.caffe.training.TrainingHelper;
import edu.brown.cs.atari_vision.caffe.vfa.DQN;
import edu.brown.cs.atari_vision.caffe.vfa.StateVectorizor;
import org.bytedeco.javacpp.FloatPointer;


/**
 * Created by MelRod on 5/29/16.
 */
public class NNGridWorld {

    static final String SOLVER_FILE = "grid_world_net_solver.prototxt";

    static final boolean GUI = true;

    static ActionSet actionSet = new ActionSet(new String[]{
            GridWorldDomain.ACTION_NORTH,
            GridWorldDomain.ACTION_SOUTH,
            GridWorldDomain.ACTION_WEST,
            GridWorldDomain.ACTION_EAST});

    public GridWorldDomain gwdg;
    public OOSADomain domain;
    public RewardFunction rf;
    public TerminalFunction tf;
    public StateConditionTest goalCondition;
    public State initialState;
    public HashableStateFactory hashingFactory;
    public Environment env;

    public DQN dqn;

    public NNGridWorld(double gamma) {

        //create the domain
        gwdg = new GridWorldDomain(11, 11);
        gwdg.setMapToFourRooms();
        rf = new UniformCostRF();
        tf = new SinglePFTF(PropositionalFunction.findPF(gwdg.generatePfs(), GridWorldDomain.PF_AT_LOCATION));
        gwdg.setRf(rf);
        gwdg.setTf(tf);
        domain = gwdg.generateDomain();

        goalCondition = new TFGoalCondition(tf);

        //set up the initial state of the task
        initialState = new GridWorldState(new GridAgent(0, 0), new GridLocation(10, 10, "loc0"));

        //set up the state hashing system for tabular algorithms
        hashingFactory = new SimpleHashableStateFactory();

        //set up the environment for learners algorithms
        env = new SimulatedEnvironment(domain, initialState);

        dqn = new DQN(SOLVER_FILE, actionSet, new NNGridStateConverter(), gamma);
    }

    public static void main(String args[]) {

        double gamma = 0.99;

        NNGridWorld nnGridWorld = new NNGridWorld(gamma);

        Policy policy = new AnnealedEpsilonGreedy(nnGridWorld.dqn, 1.0, 0.1, 1000000);

        DeepQLearner deepQLearner = new DeepQLearner(nnGridWorld.domain, gamma, 50000, policy, nnGridWorld.dqn);
        deepQLearner.setExperienceReplay(new FixedSizeMemory(1000000), nnGridWorld.dqn.batchSize);

        Policy testPolicy = new EpsilonGreedy(nnGridWorld.dqn, 0.05);

        if (GUI) {
            VisualExplorer exp = new VisualExplorer(nnGridWorld.domain, nnGridWorld.env, GridWorldVisualizer.getVisualizer(nnGridWorld.gwdg.getMap()));
            exp.initGUI();
            exp.startLiveStatePolling(33);
        }

        // setup helper
        TrainingHelper helper = new SimpleTrainer(deepQLearner, nnGridWorld.dqn, testPolicy, actionSet, nnGridWorld.env);
        helper.setTotalTrainingSteps(50000000);
        helper.setTestInterval(500000);
        helper.setNumTestEpisodes(5);
        helper.setMaxEpisodeSteps(10000);

        // run helper
        helper.run();
    }

    class NNGridStateConverter implements StateVectorizor {

        @Override
        public void vectorizeState(State state, FloatPointer input) {
            GridWorldState gwState = (GridWorldState) state;

            int width = gwdg.getWidth();

            input.fill(0);

            ObjectInstance agent = gwState.object(GridWorldDomain.CLASS_AGENT);
            int x = (Integer)agent.get(GridWorldDomain.VAR_X);
            int y = (Integer)agent.get(GridWorldDomain.VAR_Y);

            input.put((long)(y*width + x), 1);
        }
    }
}
