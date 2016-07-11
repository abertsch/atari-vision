package edu.brown.cs.atari_vision.caffe;

import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
import burlap.mdp.singleagent.SADomain;
import edu.brown.cs.atari_vision.ale.burlap.ALEDomainGenerator;
import edu.brown.cs.atari_vision.ale.burlap.ALEEnvironment;
import edu.brown.cs.atari_vision.ale.burlap.action.ActionSet;
import edu.brown.cs.atari_vision.ale.io.Actions;
import edu.brown.cs.atari_vision.caffe.experiencereplay.FrameExperienceMemory;
import edu.brown.cs.atari_vision.caffe.learners.DeepQLearner;
import edu.brown.cs.atari_vision.caffe.policies.AnnealedEpsilonGreedy;
import edu.brown.cs.atari_vision.caffe.preprocess.DQNPreProcessor;
import edu.brown.cs.atari_vision.caffe.training.DQNTrainer;
import edu.brown.cs.atari_vision.caffe.training.SimpleTrainer;
import edu.brown.cs.atari_vision.caffe.training.TrainingHelper;
import edu.brown.cs.atari_vision.caffe.vfa.DQN;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.caffe;

/**
 * Created by maroderi on 6/28/16.
 */
public class RunTest {

    static final String SOLVER_FILE = "dqn_solver.prototxt";
    static final String ROM = "pong.bin";
    static final boolean GUI = true;

    static int maxHistoryLength = 4;
    static int frameSkip = 100;

    static final double epsilonStart = 1;
    static final double epsilonEnd = 0.1;
    static final int epsilonAnnealDuration = 1000000;

    static final double gamma = 0.99;

    public static void main(String[] args) {
        Loader.load(caffe.Caffe.class);

        ActionSet actionSet = Actions.pongActionSet();

        ALEDomainGenerator domGen = new ALEDomainGenerator(actionSet);
        SADomain domain = domGen.generateDomain();

        FrameExperienceMemory testExperienceMemory = new FrameExperienceMemory(maxHistoryLength, maxHistoryLength, new DQNPreProcessor(), actionSet);
        ALEEnvironment env = new ALEEnvironment(domain, testExperienceMemory, ROM, frameSkip, GUI);

        DQN dqn = new DQN(SOLVER_FILE, actionSet, testExperienceMemory, gamma);
        dqn.loadWeightsFrom("_iter_49946067.caffemodel");

        Policy testPolicy = new EpsilonGreedy(dqn, 0.05);

        // setup helper
        TrainingHelper helper = new SimpleTrainer(null, dqn, testPolicy, actionSet, env);
        helper.setNumTestEpisodes(50);
        helper.setMaxEpisodeFrames(200000);

        // run helper
        helper.runTestSet();
    }
}
