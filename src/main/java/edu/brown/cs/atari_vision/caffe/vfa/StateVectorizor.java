package edu.brown.cs.atari_vision.caffe.vfa;

import burlap.mdp.core.state.State;
import org.bytedeco.javacpp.FloatPointer;

import static org.bytedeco.javacpp.caffe.*;

/**
 * Created by MelRod on 5/27/16.
 */
public interface StateVectorizor {

    /**
     * Converts a given state to a float vector
     *
     * @param state The state to convert.
     * @param input The float vector into which to put the state vecotr.
     */
    void vectorizeState(State state, FloatPointer input);
}
