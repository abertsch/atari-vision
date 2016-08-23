package edu.brown.cs.atari_vision.caffe.training;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.RandomPolicy;
import burlap.behavior.singleagent.Episode;
import burlap.mdp.core.Action;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
import edu.brown.cs.atari_vision.ale.burlap.action.ActionSet;
import edu.brown.cs.atari_vision.caffe.learners.DeepQLearner;
import edu.brown.cs.atari_vision.caffe.vfa.DQN;

import static org.bytedeco.javacpp.caffe.*;

import java.io.*;
import java.nio.DoubleBuffer;
import java.nio.channels.Channels;
import java.nio.channels.WritableByteChannel;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 * Created by MelRod on 5/28/16.
 */
public abstract class TrainingHelper {

    DeepQLearner learner;
    DQN vfa;
    Policy testPolicy;

    Environment env;

    ActionSet actionSet;

    int maxEpisodeFrames = -1;
    int totalTrainingFrames = 10000000;

    int testInterval = 100000;
    int numTestEpisodes = 10;

    String snapshotPrefix;
    int snapshotInterval = -1;


    List<State> sampleStates;
    int numSampleStates = -1;

    int frameCounter;
    int episodeCounter;

    double highestAverageReward = Double.NEGATIVE_INFINITY;

    PrintStream out;


    public TrainingHelper(DeepQLearner learner, DQN vfa, Policy testPolicy, ActionSet actionSet, Environment env) {
        this.learner = learner;
        this.vfa = vfa;
        this.testPolicy = testPolicy;
        this.env = env;
        this.actionSet = actionSet;

        this.frameCounter = 0;
        this.episodeCounter = 0;


        try {
            String fileName = "rmsProp1";
            out = new PrintStream(new BufferedOutputStream(new FileOutputStream(fileName)));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return;
        }
    }

    public abstract void prepareForTraining();
    public abstract void prepareForTesting();

    public void setNumSampleStates(int n) {
        numSampleStates = n;
    }

    public void setTotalTrainingFrames(int n) {
        totalTrainingFrames = n;
    }

    public void setNumTestEpisodes(int n) {
        numTestEpisodes = n;
    }

    public void setTestInterval(int i) {
        testInterval = i;
    }

    public void setMaxEpisodeFrames(int f) {
        maxEpisodeFrames = f;
    }

    public void enableSnapshots(String snapshotPrefix, int snapshotInterval) {
        this.snapshotPrefix = snapshotPrefix;
        this.snapshotInterval = snapshotInterval;
    }

    public void run() {
        if (numSampleStates > 0) {
            System.out.println("Sampling random states");
            prepareForTesting();

            // Take a random sample of states
            Policy randomPolicy = new RandomPolicy(actionSet.actionTypeList());
            sampleStates = new ArrayList<>(numSampleStates);

            while (sampleStates.size() < numSampleStates) {
                // Run a random episode
                Episode ea = runEpisode(randomPolicy);
                int episodeSize = ea.numTimeSteps();

                // Random sample of unique states from the episode
                Random rng = new Random();
                int numStatesToAdd = Math.min(episodeSize, numSampleStates - sampleStates.size());
                for (int i = 0; i < numStatesToAdd; i++) {
                    sampleStates.add(ea.state(rng.nextInt(episodeSize)));
                }
            }

            querySampleQs();
        }

        int testCountDown = testInterval;
        int snapshotCountDown = snapshotInterval;

        while (frameCounter < totalTrainingFrames) {
            System.out.println(String.format("Training Episode %d at frame %d", episodeCounter, frameCounter));

            prepareForTraining();
            env.resetEnvironment();

            long startTime = System.currentTimeMillis();
            Episode ea = learner.runLearningEpisode(env, Math.min(totalTrainingFrames - frameCounter, maxEpisodeFrames));
            long endTime = System.currentTimeMillis();
            double timeInterval = (endTime - startTime)/1000.0;

            double totalReward = 0;
            for (double r : ea.rewardSequence) {
                totalReward += r;
            }
            System.out.println(String.format("Episode reward: %.2f -- %.1ffps", totalReward, ea.numTimeSteps()/timeInterval));
            System.out.println();

            frameCounter += ea.numTimeSteps();
            episodeCounter++;
            if (snapshotPrefix != null) {
                snapshotCountDown -= ea.numTimeSteps();
                if (snapshotCountDown <= 0) {
                    saveLearningState(snapshotPrefix);
                    snapshotCountDown += snapshotInterval;
                }
            }

            testCountDown -= ea.numTimeSteps();
            if (testCountDown <= 0) {
                runTestSet(frameCounter);
                testCountDown += testInterval;
            }
        }

        System.out.println("Done Training!");
    }

    public void querySampleQs() {
        double totalMaxQ = 0;
        for (State state : sampleStates) {
            FloatBlob qVals = vfa.qValuesForState(state);
            totalMaxQ += vfa.blobMax(qVals, 0);
        }

        double averageMaxQ = totalMaxQ/numSampleStates;
        System.out.println(String.format("Average Max Q-Value for sample states: %.3f", averageMaxQ));
    }

    public void runTestSet(int frameCounter) {

        prepareForTesting();

        // Test the MaxQValues of the sample states
        if (sampleStates != null) {
            querySampleQs();
        }


        // Run the test policy on test episodes
        System.out.println("Running Test Set...");
        double totalTestReward = 0;
        for (int e = 1; e <= numTestEpisodes; e++) {
            Episode ea = runEpisode(testPolicy);

            double totalReward = 0;
            for (double reward : ea.rewardSequence) {
                totalReward += reward;
            }

            System.out.println(String.format("%d: Reward = %.2f", e, totalReward));
            totalTestReward += totalReward;
        }

        double averageReward = totalTestReward/numTestEpisodes;
        highestAverageReward = Math.max(highestAverageReward, averageReward);
        System.out.println(String.format("Average Test Reward: %.2f -- highest: %.2f", averageReward, highestAverageReward));
        System.out.println();

        out.printf("Frame %d: %.2f\n", frameCounter, averageReward);
        out.flush();
    }

    public Episode runEpisode(Policy policy) {
        env.resetEnvironment();
        Episode ea = new Episode();

        int eFrameCounter = 0;
        while(!env.isInTerminalState() && (eFrameCounter < maxEpisodeFrames || maxEpisodeFrames == -1)){
            State curState = env.currentObservation();
            Action action = policy.action(curState);

            EnvironmentOutcome eo = env.executeAction(action);
            ea.transition(eo.a, eo.op, eo.r);

            eFrameCounter++;
        }

        return ea;
    }

    public void saveLearningState(String filePrefix) {

        String trainerDataFilename = filePrefix + "_trainer.data";
        HashMap<String, Object> trainerData = new HashMap<>();
        trainerData.put("frameCounter", frameCounter);
        trainerData.put("episodeCounter", episodeCounter);
        try (ObjectOutputStream objOut = new ObjectOutputStream(new FileOutputStream(trainerDataFilename))) {
            objOut.writeObject(trainerData);
        } catch (IOException e) {
            e.printStackTrace();
        }

        vfa.saveLearningState(filePrefix);
    }

    public void loadLearningState(String filePrefix, String solverStateFile) {

        String trainerDataFilename = filePrefix + "_trainer.data";
        try (ObjectInputStream objIn = new ObjectInputStream(new FileInputStream(trainerDataFilename))) {
            HashMap<String, Object> trainerData = (HashMap<String, Object>) objIn.readObject();

            this.frameCounter = (Integer)trainerData.get("frameCounter");
            this.episodeCounter = (Integer)trainerData.get("episodeCounter");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        vfa.loadLearningState(filePrefix, solverStateFile);

        learner.restartFrom(this.frameCounter);
    }
}
