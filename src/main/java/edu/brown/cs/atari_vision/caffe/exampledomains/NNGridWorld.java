package edu.brown.cs.atari_vision.caffe.exampledomains;

import burlap.behavior.functionapproximation.ParametricFunction;
import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
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
import edu.brown.cs.atari_vision.ale.burlap.action.ActionSet;
import edu.brown.cs.atari_vision.caffe.experiencereplay.FixedSizeMemory;
import edu.brown.cs.atari_vision.caffe.learners.DeepQLearner;
import edu.brown.cs.atari_vision.caffe.policies.AnnealedEpsilonGreedy;
import edu.brown.cs.atari_vision.caffe.training.TrainingHelper;
import edu.brown.cs.atari_vision.caffe.vfa.NNVFA;
import org.bytedeco.javacpp.FloatPointer;

import static org.bytedeco.javacpp.caffe.*;

/**
 * Created by MelRod on 5/29/16.
 */
public class NNGridWorld extends NNVFA {

    static final String SOLVER_FILE = "grid_world_net_solver.prototxt";

    static final String INPUT_NAME = "frames_input_layer";
    static final String FILTER_NAME = "filter_input_layer";
    static final String YS_NAME = "target_input_layer";

    static final String SNAPSHOT_FILE_NAME = "nnGridWorldSnapshot";
    static final boolean GUI = true;

    static ActionSet actionSet = new ActionSet(new String[]{
            GridWorldDomain.ACTION_NORTH,
            GridWorldDomain.ACTION_SOUTH,
            GridWorldDomain.ACTION_WEST,
            GridWorldDomain.ACTION_EAST});

    GridWorldDomain gwdg;
    OOSADomain domain;
    RewardFunction rf;
    TerminalFunction tf;
    StateConditionTest goalCondition;
    State initialState;
    HashableStateFactory hashingFactory;
    Environment env;

    public NNGridWorld() {
        super(actionSet, 0.99);

        //create the domain
        gwdg = new GridWorldDomain(11, 11);
        gwdg.setMapToFourRooms();
        rf = new UniformCostRF();
        tf = new SinglePFTF(PropositionalFunction.getPropositionalFunction(gwdg.generatePfs(), GridWorldDomain.PF_AT_LOCATION));
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

        constructNetwork();
    }

    protected NNGridWorld(NNGridWorld nnGridWorld) {
        super(nnGridWorld);
    }

    @Override
    protected void constructNetwork() {

        String platform = System.getProperty("javacpp.platform");

        SolverParameter solver_param = new SolverParameter();
        ReadProtoFromTextFileOrDie(SOLVER_FILE, solver_param);

        if (solver_param.solver_mode() == SolverParameter_SolverMode_GPU) {
            Caffe.set_mode(Caffe.GPU);
        }

        FloatSolver solver = FloatSolverRegistry.CreateSolver(solver_param);
        this.caffeNet = solver.net();

        this.inputLayer = (FloatMemoryDataLayer)caffeNet.layer_by_name(INPUT_NAME);
        this.filterLayer = (FloatMemoryDataLayer)caffeNet.layer_by_name(FILTER_NAME);
        this.yLayer = (FloatMemoryDataLayer)caffeNet.layer_by_name(YS_NAME);
    }

    @Override
    protected FloatBlob convertStateToInput(State state) {
        int width = gwdg.getWidth();
        int height = gwdg.getHeight();

        FloatPointer input = (new FloatPointer(width * height)).fill(0);

        ObjectInstance agent = ((OOState)state).object(GridWorldDomain.CLASS_AGENT);
        int x = (Integer)agent.get(GridWorldDomain.VAR_X);
        int y = (Integer)agent.get(GridWorldDomain.VAR_Y);

        input.put((long)(y*width + x), 1);

        return new FloatBlob(input);
    }

    @Override
    protected int inputSize() {
        int width = gwdg.getWidth();
        int height = gwdg.getHeight();

        return width * height;
    }

    @Override
    public ParametricFunction copy() {
        return new NNGridWorld(this);
    }


    public static void main(String args[]) {

        NNGridWorld nnGridWorld = new NNGridWorld();

        if (GUI) {
            VisualExplorer exp = new VisualExplorer(nnGridWorld.domain, nnGridWorld.env, GridWorldVisualizer.getVisualizer(nnGridWorld.gwdg.getMap()));
            exp.initGUI();
            exp.startLiveStatePolling(10);
        }


        Policy policy = new AnnealedEpsilonGreedy(nnGridWorld, 1.0, 0.1, 1000000);

        DeepQLearner deepQLearner = new DeepQLearner(nnGridWorld.domain, 0.99, policy, nnGridWorld);
        deepQLearner.setExperienceReplay(new FixedSizeMemory(1000000), 32);

        Policy testPolicy = new EpsilonGreedy(nnGridWorld, 0.05);

        // setup helper
        TrainingHelper helper = new TrainingHelper(deepQLearner, nnGridWorld, testPolicy, actionSet, nnGridWorld.env);
        helper.setTotalTrainingFrames(10000000);
        helper.setTestInterval(100000);
        helper.setNumTestEpisodes(10);
        helper.setNumSampleStates(1000);
        helper.setSnapshots(SNAPSHOT_FILE_NAME, 50000);

        // run helper
        helper.run();
    }
}