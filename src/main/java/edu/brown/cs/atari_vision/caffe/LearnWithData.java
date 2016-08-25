package edu.brown.cs.atari_vision.caffe;

import burlap.behavior.functionapproximation.ParametricFunction;
import burlap.behavior.policy.EpsilonGreedy;
import burlap.behavior.policy.Policy;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
import edu.brown.cs.atari_vision.ale.burlap.ALEDomainGenerator;
import edu.brown.cs.atari_vision.ale.burlap.ALEEnvironment;
import edu.brown.cs.atari_vision.ale.burlap.action.ActionSet;
import edu.brown.cs.atari_vision.ale.io.Actions;
import edu.brown.cs.atari_vision.caffe.experiencereplay.FrameExperienceMemory;
import edu.brown.cs.atari_vision.caffe.learners.DeepQLearner;
import edu.brown.cs.atari_vision.caffe.preprocess.DQNPreProcessor;
import edu.brown.cs.atari_vision.caffe.training.SimpleTrainer;
import edu.brown.cs.atari_vision.caffe.training.TrainingHelper;
import edu.brown.cs.atari_vision.caffe.vfa.DQN;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.caffe;
import org.junit.Assert;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.caffe.*;

/**
 * Created by maroderi on 7/25/16.
 */
public class LearnWithData {

    static final String SOLVER_FILE = "dqn_solver.prototxt";
    static final String ROM = "breakout.bin";

    static final int staleDuration = 10000/4;
    static DQN staleDQN;
    static DQN dqn;

    public static void main(String[] args) {
        Loader.load(caffe.Caffe.class);
        String dataDir = "/home/maroderi/projects/data/dqn_weight_data";

        boolean usingStale = true;
        boolean checking = false;

        int sampleSize = 32;
        int numActions = 4;
        int outputInterval = 10000;

        // Create DQN
        ActionSet actionSet = Actions.breakoutActionSet();

        ALEDomainGenerator domGen = new ALEDomainGenerator(actionSet);
        SADomain domain = domGen.generateDomain();

        FrameExperienceMemory experienceMemory = new FrameExperienceMemory(1000, 4, new DQNPreProcessor(), actionSet);
        ALEEnvironment env = new ALEEnvironment(domain, experienceMemory, ROM, 4, true);

        dqn = new DQN(SOLVER_FILE, actionSet, experienceMemory, 0.99);
        DQN testDQN = new DQN(SOLVER_FILE, actionSet, experienceMemory, 0.99);
        String sampleDir0 = String.format("%s/%d", dataDir, 0);
        setNetworkWeights(dqn.caffeNet, String.format("%s/%s", sampleDir0, "weights"));

        staleDQN = new DQN(SOLVER_FILE, actionSet, experienceMemory, 0.99);// (DQN) dqn.copy();
        setNetworkWeights(staleDQN.caffeNet, String.format("%s/%s", sampleDir0, "weights"));

        DQNPreProcessor preProcessor = new DQNPreProcessor();

        int testInterval = 250000/4;
        Policy testPolicy = new EpsilonGreedy(dqn, 0.05);
        // setup helper
        TrainingHelper helper = new SimpleTrainer(null, dqn, testPolicy, actionSet, env);
        helper.setNumTestEpisodes(50);
        helper.setMaxEpisodeFrames(200000);

        long time = System.currentTimeMillis();
        for (int s = 0; true; s++) {
            if (usingStale) {
                if (s % staleDuration == 0) {
                    updateStaleFunction();
                }
            }

            String sampleDir = String.format("%s/%d", dataDir, s);

//            if (s % 1 == 0 || (s > 400)) {
//                System.out.println("---------------------");
//                setNetworkWeights(testDQN.caffeNet, String.format("%s/%s", sampleDir, "weights"));
//                verifyNetworkWeights(dqn.caffeNet, testDQN.caffeNet, 0.00001f, s);
//            }

            String statesFile = String.format("%s/%s", sampleDir, "s.png");

//            setNetworkWeights(dqn.caffeNet, String.format("%s/%s", sampleDir, "weights"));

            preProcessor.convertDataToInput(loadImageData(statesFile).data(), dqn.stateInputs, 32*4);

            FloatPointer ys = (new FloatPointer(sampleSize * numActions)).zero();
            FloatPointer actionFilter = (new FloatPointer(sampleSize * numActions)).zero();

            String actionFile = String.format("%s/%s", sampleDir, "a");
            int[] actions = readIntData(actionFile, ",");
            String rFile = String.format("%s/%s", sampleDir, "r");
            int[] r = readIntData(rFile, ",");
            String tFile = String.format("%s/%s", sampleDir, "term");
            int[] t = readIntData(tFile, ",");

            float[] maxQs;
            if (usingStale) {
                // if using stale
                String primeStatesFile = String.format("%s/%s", sampleDir, "s2.png");
                preProcessor.convertDataToInput(loadImageData(primeStatesFile).data(), dqn.primeStateInputs, 32*4);
                staleDQN.inputDataIntoLayers(dqn.primeStateInputs.position(0), dqn.dummyInputData, dqn.dummyInputData);
                staleDQN.caffeNet.ForwardPrefilled();

                maxQs = new float[sampleSize];
                for (int i = 0; i < sampleSize; i++) {
                    maxQs[i] = dqn.blobMax(staleDQN.qValuesBlob, i);
                }
            } else {
                // if not using stale
                String maxQFile = String.format("%s/%s", sampleDir, "q2_max");
                maxQs = readFloatData(maxQFile, ",");
            }

            // For getting the values for clipping
            dqn.inputDataIntoLayers(dqn.stateInputs.position(0), dqn.dummyInputData, dqn.dummyInputData);
            dqn.caffeNet.ForwardPrefilled();

            for (int i = 0; i < sampleSize; i++) {
                int a = actions[i] - 1; // because Lua is 1-indexed
                int index = i*numActions + a;

                float y;
                if (t[i] == 0) {
                    y = r[i] + (float)dqn.gamma * maxQs[i];
                } else {
                    y = r[i];
                }

                float q = dqn.qValuesBlob.data_at(i, a, 0, 0);
                float delta_clip = 1;
                if (y - q > delta_clip) {
                    y = q + delta_clip;
                } else if (y - q < -delta_clip) {
                    y = q - delta_clip;
                }

                ys.put(index, y);
                actionFilter.put(index, 1);
            }

            // Backprop
            dqn.inputDataIntoLayers(dqn.stateInputs.position(0), actionFilter, ys);
            dqn.caffeNet.ForwardPrefilled();
            dqn.caffeNet.Backward();

            // Check weights
//            String qFile = String.format("%s/%s", sampleDir, "q");
//            float[] q = readFloatData(qFile);

//            System.out.println("----------");
//            Debug.printBlob(dqn.qValuesBlob);
//            System.out.println("-------------");

            dqn.caffeSolver.Step(1);

            if (checking) {
                if (s % 1 == 0 || (s > 400)) {
                    testDQN.caffeNet.CopyTrainedLayersFrom(String.format("_iter_%d.caffemodel", s+1));
                    verifyNetworkWeights(dqn.caffeNet, testDQN.caffeNet, 0, s);
                }
            } else {
                dqn.caffeSolver.Snapshot();
            }

            FloatBlob outBlob = dqn.caffeNet.blob_by_name("states");
            if (s % outputInterval == 0) {
                long currentTime = System.currentTimeMillis();
                float fps = outputInterval/((currentTime - time)/1000f);
                System.out.printf("%d: %.2f steps/sec\n", s, fps);

                time = currentTime;
            }

//            if (s % 1000 == 0) {
//                float[] qs = new float[sampleSize*numActions];
//                int i = 0;
//                for (int n = 0; n < sampleSize; n++) {
//                    for (int c = 0; c < numActions; c++) {
//                        qs[i++] = dqn.qValuesBlob.data_at(n,c,0,0);
//                    }
//                }
//                System.out.println("DONE!");
//            }

            // run test set
            if (s % testInterval == 0) {
//                helper.runTestSet(s);
            }
        }
    }

    public static void updateStaleFunction() {
        staleDQN.updateParamsToMatch(dqn);
    }

    public static Mat loadImageData(String fileName) {
        try {
            BufferedImage image = ImageIO.read(new File(fileName));
            byte[] data = ((DataBufferByte) image.getData().getDataBuffer()).getData();
            return new Mat(data);
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static String[] readStringData(String fileName, String split) {
        String line;
        try {
            BufferedReader in = new BufferedReader(new FileReader(fileName));
            line = in.readLine();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

        return line.split(split);
    }

    public static String[] readStringLines(String fileName) {
        ArrayList<String> lines = new ArrayList<>();
        String line;
        try {
            BufferedReader in = new BufferedReader(new FileReader(fileName));
            while ((line = in.readLine()) != null) {
                lines.add(line);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }

        return lines.toArray(new String[0]);
    }

    public static int[] readIntData(String fileName, String split) {
        String[] stringData = readStringData(fileName, split);
        int[] data = new int[stringData.length];
        for (int i = 0; i < stringData.length; i++) {
            data[i] = Integer.parseInt(stringData[i]);
        }
        return data;
    }

    public static float[] readFloatData(String fileName, String split) {
        String[] stringData = readStringData(fileName, split);
        float[] data = new float[stringData.length];
        for (int i = 0; i < stringData.length; i++) {
            data[i] = Float.parseFloat(stringData[i]);
        }
        return data;
    }

    public static void setNetworkWeights(FloatNet net, String weightFile) {

        FloatBlobSharedVector params = net.params();

        String[] layerParams = readStringLines(weightFile);

        String[] stringData = layerParams[22].split(" ");
        float[] data = new float[stringData.length];
        for (int i = 0; i < stringData.length; i++) {
            data[i] = Float.valueOf(stringData[i]);
        }

        int index = 0;
        for (int layer = 0; layer < 10; layer++) {
            FloatBlob blob = params.get(layer);
//            System.out.println(String.format("%d: %d,%d,%d,%d", layer, blob.num(), blob.channels(), blob.height(), blob.width()));
            FloatPointer blobData = blob.cpu_data();

            int size = blob.count();
            blobData.put(data, index, size);
            index += size;
        }

//        int len = data.length;
//        System.out.printf("%f,%f,%f,%f", data[len-4], data[len-3], data[len-2], data[len-1]);
    }

//    public static void setNetworkWeights(FloatNet net, String weightDir) {
//
//        FloatBlobSharedVector params = net.params();
//
//        for (int i = 0; i < params.size(); i++) {
//            String fileName = String.format("%s/%d", weightDir, i+1);
//            float[] data = readFloatData(fileName);
//
//            FloatBlob blob = params.get(i);
//            System.out.println(String.format("%d: %d,%d,%d,%d", i, blob.num(), blob.channels(), blob.height(), blob.width()));
//            FloatPointer blobData = blob.cpu_data();
//
//            blobData.put(data);
////            params.get(i).set_cpu_data(data)
//
//        }
//
//        verifyNetworkWeights(net, weightDir);
//    }

    public static void verifyNetworkWeights(FloatNet net, FloatNet testNet, float delta, int step) {
        FloatBlobSharedVector params = net.params();
        FloatBlobSharedVector testParams = testNet.params();

        float max = 0;
        int[] coords = new int[5];
        for (int i = 0; i < params.size(); i++) {
            FloatBlob blob = params.get(i);
            FloatBlob testBlob = testParams.get(i);

            for (int n = 0; n < blob.num(); n++) {
                for (int c = 0; c < blob.channels(); c++) {
                    for (int h = 0; h < blob.height(); h++) {
                        for (int w = 0; w < blob.width(); w++) {
                            float err = blob.data_at(n, c, h, w) - testBlob.data_at(n, c, h, w);
                            if (max < err) {
                                max = err;
                                coords[0] = i;
                                coords[1] = n;
                                coords[2] = c;
                                coords[3] = h;
                                coords[4] = w;
                            }
                            if (Math.abs(err) > delta) {
//                                System.out.printf("NOT EQUAL: layer %d: %d,%d,%d,%d\n", i, n,c,h,w);
                            }
                        }
                    }
                }
            }
        }

        System.out.printf("%d: Delta %.10f, Max %.10f -- %d,%d,%d,%d,%d\n", step, delta, max, coords[0],coords[1],coords[2],coords[3],coords[4]);
    }
}
