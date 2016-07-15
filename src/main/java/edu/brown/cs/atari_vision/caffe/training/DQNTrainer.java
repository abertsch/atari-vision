package edu.brown.cs.atari_vision.caffe.training;

import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.Environment;
import edu.brown.cs.atari_vision.caffe.action.ActionSet;
import edu.brown.cs.atari_vision.caffe.experiencereplay.FrameExperienceMemory;
import edu.brown.cs.atari_vision.caffe.learners.DeepQLearner;
import edu.brown.cs.atari_vision.caffe.policies.AnnealedEpsilonGreedy;
import edu.brown.cs.atari_vision.caffe.preprocess.ALEPreProcessor;
import edu.brown.cs.atari_vision.caffe.vfa.DQN;
import edu.brown.cs.burlap.ALEDomainGenerator;
import edu.brown.cs.burlap.ALEEnvironment;
import edu.brown.cs.burlap.gui.ALEVisualExplorer;
import edu.brown.cs.burlap.gui.ALEVisualizer;
import org.bytedeco.javacpp.Loader;

import static org.bytedeco.javacpp.caffe.*;

/**
 * Created by MelRod on 5/31/16.
 */
public class DQNTrainer extends TrainingHelper {

    static final String SOLVER_FILE = "dqn_solver.prototxt";
    static final String alePath = "/home/maroderi/projects/Arcade-Learning-Environment/ale";
    static final String ROM = "/home/maroderi/projects/atari_roms/breakout.bin";
    static final boolean GUI = true;

    static final int experienceMemoryLength = 1000000;
    static int maxHistoryLength = 4;
    static int frameSkip = 4;

    static final double epsilonStart = 1;
    static final double epsilonEnd = 0.1;
    static final int epsilonAnnealDuration = 1000000;

    static final double gamma = 0.99;

    static final String[] actionNames = {
            "player_a_noop", "player_a_right", "player_a_left", "player_a_fire"
    };


    protected FrameExperienceMemory trainingMemory;
    protected FrameExperienceMemory testMemory;

    public DQNTrainer(DeepQLearner learner, DQN vfa, Policy testPolicy, ActionSet actionSet, Environment env,
                      FrameExperienceMemory trainingMemory,
                      FrameExperienceMemory testMemory) {
        super(learner, vfa, testPolicy, actionSet, env);

        this.trainingMemory = trainingMemory;
        this.testMemory = testMemory;
    }

    @Override
    public void prepareForTraining() {
        vfa.stateConverter = trainingMemory;
    }

    @Override
    public void prepareForTesting() {
        vfa.stateConverter = testMemory;
    }

    public static void main(String[] args) {

        Loader.load(Caffe.class);

        ALEDomainGenerator domGen = new ALEDomainGenerator(actionNames);
        SADomain domain = domGen.generateDomain();

        ActionSet actionSet = new ActionSet(domain);

        FrameExperienceMemory trainingExperienceMemory =
                new FrameExperienceMemory(experienceMemoryLength, maxHistoryLength, new ALEPreProcessor(), actionSet);
        ALEEnvironment env = new ALEEnvironment(alePath, ROM, frameSkip);
        ALEVisualExplorer exp = new ALEVisualExplorer(domain, env, ALEVisualizer.create());
        exp.initGUI();
        exp.startLiveStatePolling(1000/60);

        FrameExperienceMemory testExperienceMemory = new FrameExperienceMemory(10000, maxHistoryLength, new ALEPreProcessor(), actionSet);

        DQN dqn = new DQN(SOLVER_FILE, actionSet, trainingExperienceMemory, gamma);
        Policy policy = new AnnealedEpsilonGreedy(dqn, epsilonStart, epsilonEnd, epsilonAnnealDuration);

        DeepQLearner deepQLearner = new DeepQLearner(domain, gamma, 50000, policy, dqn, trainingExperienceMemory);
        deepQLearner.setExperienceReplay(trainingExperienceMemory, dqn.batchSize);

        Policy testPolicy = new EpsilonGreedy(dqn, 0.05);

        // setup helper
        TrainingHelper helper =
                new DQNTrainer(deepQLearner, dqn, testPolicy, actionSet, env, trainingExperienceMemory, testExperienceMemory);
        helper.setTotalTrainingSteps(50000000);
        helper.setTestInterval(100000);
        helper.setNumTestEpisodes(10);
        helper.setMaxEpisodeSteps(20000);
        helper.enableSnapshots("networks/dqn/breakout_temp", 1000000);

        // load learning state if resuming
//        helper.loadLearningState("networks/dqn/pong");

        // run helper
        helper.run();
    }
}
