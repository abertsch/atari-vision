package edu.brown.cs.atari_vision;

import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.RandomPolicy;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
import edu.brown.cs.atari_vision.ale.burlap.ALEDomainGenerator;
import edu.brown.cs.atari_vision.ale.burlap.ALEEnvironment;
import edu.brown.cs.atari_vision.ale.burlap.action.ActionSet;
import edu.brown.cs.atari_vision.ale.io.Actions;
import edu.brown.cs.atari_vision.caffe.Debug;
import edu.brown.cs.atari_vision.caffe.experiencereplay.FrameExperience;
import edu.brown.cs.atari_vision.caffe.experiencereplay.FrameExperienceMemory;
import edu.brown.cs.atari_vision.caffe.experiencereplay.FrameHistoryState;
import edu.brown.cs.atari_vision.caffe.learners.DeepQLearner;
import edu.brown.cs.atari_vision.caffe.policies.AnnealedEpsilonGreedy;
import edu.brown.cs.atari_vision.caffe.preprocess.DQNPreProcessor;
import edu.brown.cs.atari_vision.caffe.training.DQNTrainer;
import edu.brown.cs.atari_vision.caffe.training.TrainingHelper;
import edu.brown.cs.atari_vision.caffe.vfa.DQN;
import org.bytedeco.javacpp.*;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Created by maroderi on 7/21/16.
 */
public class FrameHistoryTest {

    ALEEnvironment env;
    FrameExperienceMemory experienceMemory;
    Policy randomPolicy;
    int maxHistoryLength = 4;

    @Before
    public void setup() {

        String romPath = "/home/maroderi/projects/atari_roms/breakout.bin";
        String alePath = "/home/maroderi/projects/Arcade-Learning-Environment/ale";

        int experienceMemoryLength = 1000;

        Loader.load(caffe.Caffe.class);

        ActionSet actionSet = Actions.breakoutActionSet();

        ALEDomainGenerator domGen = new ALEDomainGenerator(actionSet);
        SADomain domain = domGen.generateDomain();

        experienceMemory = new FrameExperienceMemory(experienceMemoryLength, maxHistoryLength, new DQNPreProcessor(), actionSet);
        env = new ALEEnvironment(domain, experienceMemory, "breakout.bin", 4, true);

        randomPolicy = new RandomPolicy(domain);
    }

    @After
    public void teardown() {

    }

    @Test
    public void Test() {
        List<Long> indexList = new ArrayList<>();
        List<EnvironmentOutcome> eoList = new ArrayList<>();
        List<FloatPointer> dataList = new ArrayList<>();

        for (int step = 0; step < experienceMemory.experiences.length*4; step++) {
            State s = env.currentObservation();
            EnvironmentOutcome eo = env.executeAction(randomPolicy.action(s));

            FrameHistoryState frameHistoryState = (FrameHistoryState) eo.op;
            FloatPointer data = new FloatPointer(experienceMemory.preProcessor.outputSize() * maxHistoryLength);

            experienceMemory.getStateInput(frameHistoryState, data);

            indexList.add(frameHistoryState.index);
            eoList.add(eo);
            dataList.add(data);

            if (eo.terminated) {
                env.resetEnvironment();
            }
        }

        List<EnvironmentOutcome> samples = experienceMemory.sampleExperiences(experienceMemory.experiences.length + 1);
        FloatPointer input = new FloatPointer(experienceMemory.preProcessor.outputSize() * maxHistoryLength);
        int s = 0;
        for (EnvironmentOutcome sample : samples) {
            int listIndex = -1;
            for (int i = indexList.size() - 1; i >= experienceMemory.experiences.length; i--) {
                FrameHistoryState frameHistoryState = (FrameHistoryState) sample.op;
                if (frameHistoryState.index == indexList.get(i)) {
                    listIndex = i;
                    break;
                }
            }
            Assert.assertTrue(listIndex != -1);

            EnvironmentOutcome eo = eoList.get(listIndex);
            FloatPointer data = dataList.get(listIndex);

            Assert.assertEquals(eo.a.actionName(), sample.a.actionName());
            Assert.assertEquals(eo.r, sample.r, 0.001);
            Assert.assertEquals(((FrameHistoryState) eo.o).index, ((FrameHistoryState) sample.o).index);
            Assert.assertEquals(((FrameHistoryState) eo.o).historyLength, ((FrameHistoryState) sample.o).historyLength);
            Assert.assertEquals(((FrameHistoryState) eo.op).index, ((FrameHistoryState) sample.op).index);
            Assert.assertEquals(((FrameHistoryState) eo.op).historyLength, ((FrameHistoryState) sample.op).historyLength);

            // Check the same image
            long size = experienceMemory.preProcessor.outputSize()*maxHistoryLength;
            FloatBuffer buf1 = data.asBuffer();
            experienceMemory.getStateInput((FrameHistoryState)sample.op, input.position(0));
            FloatBuffer buf2 = input.asBuffer();

            s++;
            for (int b = 0; b < size; b++) {
                Assert.assertEquals(buf1.get(), buf2.get(), 0.0000001);
            }
        }
    }
}
