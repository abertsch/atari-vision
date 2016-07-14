package edu.brown.cs.atari_vision.caffe.vfa;

import burlap.mdp.core.state.State;
import org.bytedeco.javacpp.FloatPointer;

import static org.bytedeco.javacpp.caffe.*;

/**
 * Created by MelRod on 5/27/16.
 */
public interface NNStateConverter {

    /**
     * Converts a given state to Caffe input data (a float vector)
     *
     * @param state The state to convert.
     * @param input The float vector into which to put the input data.
     */
    void convertState(State state, FloatPointer input);
}
