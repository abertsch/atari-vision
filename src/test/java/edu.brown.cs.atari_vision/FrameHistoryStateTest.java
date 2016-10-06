package edu.brown.cs.atari_vision;

import burlap.mdp.core.action.Action;
import edu.brown.cs.atari_vision.ale.burlap.action.ActionSet;
import edu.brown.cs.atari_vision.caffe.experiencereplay.FrameExperienceMemory;
import edu.brown.cs.atari_vision.caffe.experiencereplay.FrameHistoryState;
import edu.brown.cs.atari_vision.caffe.preprocess.PreProcessor;
import edu.brown.cs.atari_vision.caffe.vfa.NNStateConverter;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;

import static org.bytedeco.javacpp.opencv_core.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_core;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by maroderi on 6/16/16.
 */
public class FrameHistoryStateTest {

    FloatPointer input;


    @Before
    public void setup() {
        Loader.load(opencv_core.class);
    }

    @After
    public void teardown() {

    }

    @Test
    public void TestSmall() {
        BytePointer data0 = new BytePointer((byte)0, (byte)0);
        BytePointer data1 = new BytePointer((byte)0, (byte)1);
        BytePointer data2 = new BytePointer((byte)2, (byte)3);
        BytePointer data3 = new BytePointer((byte)4, (byte)5);
        BytePointer data4 = new BytePointer((byte)6, (byte)7);
        BytePointer data5 = new BytePointer((byte)8, (byte)9);
        BytePointer data6 = new BytePointer((byte)10, (byte)11);
        BytePointer data7 = new BytePointer((byte)12, (byte)13);

        Mat frame0 = new Mat(1, 2, CV_8U, data0);
        Mat frame1 = new Mat(1, 2, CV_8U, data1);
        Mat frame2 = new Mat(1, 2, CV_8U, data2);
        Mat frame3 = new Mat(1, 2, CV_8U, data3);
        Mat frame4 = new Mat(1, 2, CV_8U, data4);
        Mat frame5 = new Mat(1, 2, CV_8U, data5);
        Mat frame6 = new Mat(1, 2, CV_8U, data6);
        Mat frame7 = new Mat(1, 2, CV_8U, data7);


        input = new FloatPointer(2 * 2);

        ActionSet actionSet = new ActionSet(new String[]{"Action0"});
        Action action0 = actionSet.getAction(0);

        FrameExperienceMemory experienceMemory = new FrameExperienceMemory(5, 2, new TestPreprocessor(2), actionSet);
        FrameHistoryState state0 = experienceMemory.initialState(frame0);
        FrameHistoryState state1 = experienceMemory.nextState(frame1, state0, action0, 0, false);
        FrameHistoryState state2 = experienceMemory.nextState(frame2, state1, action0, 0, false);
        FrameHistoryState state3 = experienceMemory.nextState(frame3, state2, action0, 0, false);
        FrameHistoryState state4 = experienceMemory.nextState(frame4, state3, action0, 0, false);

        compare(state0, experienceMemory, new BytePointer[]{data0, data0}, 2);
        compare(state1, experienceMemory, new BytePointer[]{data0, data1}, 2);
        compare(state2, experienceMemory, new BytePointer[]{data1, data2}, 2);
        compare(state3, experienceMemory, new BytePointer[]{data2, data3}, 2);
        compare(state4, experienceMemory, new BytePointer[]{data3, data4}, 2);

        FrameHistoryState state5 = experienceMemory.nextState(frame5, state4, action0, 0, false);
        FrameHistoryState state6 = experienceMemory.nextState(frame6, state5, action0, 0, false);
        FrameHistoryState state7 = experienceMemory.nextState(frame7, state6, action0, 0, false);

        compare(state3, experienceMemory, new BytePointer[]{data2, data3}, 2);
        compare(state4, experienceMemory, new BytePointer[]{data3, data4}, 2);
        compare(state5, experienceMemory, new BytePointer[]{data4, data5}, 2);
        compare(state6, experienceMemory, new BytePointer[]{data5, data6}, 2);
        compare(state7, experienceMemory, new BytePointer[]{data6, data7}, 2);
    }

    @Test
    public void TestRandom() {
        int replaySize = 50;
        int history = 4;
        int frameSize = 10;
        input = new FloatPointer(frameSize * history);

        ActionSet actionSet = new ActionSet(new String[]{"Action0"});
        Action action0 = actionSet.getAction(0);

        FrameExperienceMemory experienceMemory = new FrameExperienceMemory(replaySize, history, new TestPreprocessor(frameSize), actionSet);
        BytePointer data0 = new BytePointer(history);
        for (int f = 0; f < frameSize; f++) {
            data0.position(f).put((byte)0);
        }
        data0.position(0);
        List<BytePointer> dataList = new ArrayList<>();
        for (int h = 0; h < history; h++) {
            dataList.add(data0);
        }

        List<List<BytePointer>> dataListList = new ArrayList<>();

        List<FrameHistoryState> states = new ArrayList<>();

        FrameHistoryState prevState = null;

        for (int n = 0; n < 100; n++) {
            for (int i = 0; i < replaySize; i++) {
                BytePointer data = new BytePointer(frameSize);
                for (int f = 0; f < frameSize; f++) {
                    byte d = (byte) (Math.random()*126.0);
                    data.position(f).put(d);
                }
                data.position(0);
                dataList.remove(0);
                dataList.add(data);

                Mat frame = new Mat(1, frameSize, CV_8U, data);

                FrameHistoryState state;
                if (prevState == null) {
                    state = experienceMemory.initialState(frame);
                } else {
                    state = experienceMemory.nextState(frame, prevState, action0, 0, false);
                }
                prevState = state;

                compare(state, experienceMemory, dataList.toArray(new BytePointer[history]), frameSize);

                if (i < dataListList.size()) {
                    dataListList.set(i, new ArrayList<>(dataList));
                    states.set(i, state);
                } else {
                    dataListList.add(new ArrayList<>(dataList));
                    states.add(state);
                }

                for (int k = 0; k < states.size(); k++) {
                    compare(states.get(k), experienceMemory, dataListList.get(k).toArray(new BytePointer[history]), frameSize);
                }
            }
        }
    }

    public void compare(FrameHistoryState state, NNStateConverter<FrameHistoryState> stateConverter, BytePointer[]dataArray, long outputSize) {
        stateConverter.getStateInput(state, input);

        int i = 0;
        for (BytePointer data : dataArray) {
            for (int k = 0; k < outputSize; k++) {
                Assert.assertEquals(input.get(i), data.get(k), 1e-6);
                i++;
            }
        }
    }


    public class TestPreprocessor implements PreProcessor {
        int frameSize;

        public TestPreprocessor(int frameSize) {
            this.frameSize = frameSize;
        }

        @Override
        public BytePointer convertScreenToData(Mat screen) {
            return screen.data();
        }

        @Override
        public void convertDataToInput(BytePointer data, FloatPointer input, long size) {
            int dataSize = outputSize() * (int)size;

            Mat mat = new Mat(1, dataSize, CV_8U, data);
            Mat floatMat = new Mat(1, dataSize, CV_32F, (new BytePointer(input)).position(input.position() * input.sizeof()));

            mat.convertTo(floatMat, CV_32F, 1, 0);
        }

        @Override
        public int outputSize() {
            return frameSize;
        }
    }
}
