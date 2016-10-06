package edu.brown.cs.atari_vision.caffe.training;

import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.policy.RandomPolicy;
import burlap.behavior.singleagent.Episode;

import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.SimpleAction;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
import edu.brown.cs.atari_vision.ale.burlap.ALEEnvironment;
import edu.brown.cs.atari_vision.ale.burlap.action.ActionSet;
import edu.brown.cs.atari_vision.caffe.LearnWithData2;
import edu.brown.cs.atari_vision.caffe.learners.DeepQLearner;
import edu.brown.cs.atari_vision.caffe.vfa.DQN;

import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;
import static org.bytedeco.javacpp.caffe.*;

import java.io.*;
import java.nio.DoubleBuffer;
import java.nio.channels.Channels;
import java.nio.channels.WritableByteChannel;
import java.nio.file.Files;
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
    int totalTestSteps = 10;

    String snapshotPrefix;
    int snapshotInterval = -1;

    int frameCounter;
    int episodeCounter;

    double highestAverageReward = Double.NEGATIVE_INFINITY;
    double lowestAverageReward = Double.POSITIVE_INFINITY;

    Random rng = new Random();

    public String resultsPrefix;
    protected PrintStream testOutput;



    public TrainingHelper(DeepQLearner learner, DQN vfa, Policy testPolicy, ActionSet actionSet, Environment env) {
        this.learner = learner;
        this.vfa = vfa;
        this.testPolicy = testPolicy;
        this.env = env;
        this.actionSet = actionSet;

        this.frameCounter = 0;
        this.episodeCounter = 0;
    }

    public abstract void prepareForTraining();
    public abstract void prepareForTesting();

    public void setTotalTestSteps(int n) {
        totalTestSteps = n;
    }

    public void setTotalTrainingFrames(int n) {
        totalTrainingFrames = n;
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

    public void recordResultsTo(String resultsPrefix) {
        File dir = new File(resultsPrefix);
        if (!dir.exists() && !dir.mkdirs()) {
            throw new RuntimeException(String.format("Could not create the directory: %s", resultsPrefix));
        }

        this.resultsPrefix = resultsPrefix;

        try {
            String fileName = new File(resultsPrefix, "testResults").toString();
            testOutput = new PrintStream(new BufferedOutputStream(new FileOutputStream(fileName)));
        } catch (FileNotFoundException e) {
            e.printStackTrace();

            throw new RuntimeException(String.format("Can't open %s", resultsPrefix));
        }
    }

    public void run() {

        int testCountDown = testInterval;
        int snapshotCountDown = snapshotInterval;

        while (frameCounter < totalTrainingFrames) {
            System.out.println(String.format("Training Episode %d at frame %d", episodeCounter, frameCounter));

            prepareForTraining();
            env.resetEnvironment();
            for (int i = rng.nextInt(32); i >= 1; i--) {
                ((ALEEnvironment) env).io.act(0);
            }

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

//    public void querySampleQs() {
//        double totalMaxQ = 0;
//        for (State state : sampleStates) {
//            FloatBlob qVals = vfa.qValuesForState(state);
//            totalMaxQ += vfa.blobMax(qVals, 0);
//        }
//
//        double averageMaxQ = totalMaxQ/numSampleStates;
//        System.out.println(String.format("Average Max Q-Value for sample states: %.3f", averageMaxQ));
//    }

    public void runTestSet(int frameCounter) {

        prepareForTesting();

        // Run the test policy on test episodes
        System.out.println("Running Test Set...");
        int numSteps = 0;
        int numEpisodes = 0;
        double totalTestReward = 0;
        while (true) {
            env.resetEnvironment();
            Episode e = PolicyUtils.rollout(testPolicy, env, Math.min(maxEpisodeFrames, totalTestSteps - numSteps));

            double totalReward = 0;
            for (double reward : e.rewardSequence) {
                totalReward += reward;
            }

            System.out.println(String.format("%d: Reward = %.2f, Steps = %d", numEpisodes, totalReward, numSteps));

            numSteps += e.numTimeSteps();
            if (numSteps >= totalTestSteps) {
                if (numEpisodes == 0) {
                    totalTestReward = totalReward;
                    numEpisodes = 1;
                }
                break;
            }

            totalTestReward += totalReward;
            numEpisodes += 1;
        }

        double averageReward = totalTestReward/numEpisodes;
        if (averageReward > highestAverageReward) {
            if (resultsPrefix != null) {
                snapshot(new File(resultsPrefix, "best_net.caffemodel").toString(),  null);
            }
            highestAverageReward = averageReward;
        }
        if (frameCounter > 30000000 && averageReward < lowestAverageReward) {
            if (resultsPrefix != null) {
                snapshot(new File(resultsPrefix, "worst_net.caffemodel").toString(),  null);
            }
            lowestAverageReward = averageReward;
        }

        System.out.println(String.format("Average Test Reward: %.2f -- highest: %.2f", averageReward, highestAverageReward));
        System.out.println();

        testOutput.printf("Frame %d: %.2f\n", frameCounter, averageReward);
        testOutput.flush();
    }

    public void snapshot(String modelFileName, String solverFileName) {
        vfa.caffeSolver.Snapshot();

        // get file names
        int iter = vfa.caffeSolver.iter();
        String modelFile = String.format("_iter_%d.caffemodel", iter);
        String solverFile = String.format("_iter_%d.solverstate", iter);

        // move Caffe files
        try {
            if (modelFileName != null) {
                Files.move(Paths.get(modelFile), Paths.get(modelFileName), REPLACE_EXISTING);
            } else {
                Files.delete(Paths.get(modelFile));
            }

            if (solverFileName != null) {
                Files.move(Paths.get(solverFile), Paths.get(solverFileName), REPLACE_EXISTING);
            } else {
                Files.delete(Paths.get(solverFile));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
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
