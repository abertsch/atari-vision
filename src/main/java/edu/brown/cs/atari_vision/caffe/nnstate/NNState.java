package edu.brown.cs.atari_vision.caffe.nnstate;

import edu.brown.cs.atari_vision.ale.burlap.ALEState;

import static org.bytedeco.javacpp.caffe.*;

/**
 * Created by MelRod on 5/27/16.
 */
public interface NNState extends ALEState {

    public FloatBlob getInput();
}