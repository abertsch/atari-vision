package edu.brown.cs.atari_vision.caffe.experiencereplay;

import burlap.mdp.core.state.State;

import java.io.Serializable;
import java.util.List;

/**
 * Created by MelRod on 5/27/16.
 */
public class FrameHistoryState implements State, Serializable {

    public long index;
    public int historyLength;


    public FrameHistoryState(long index, int historyLength) {
        this.index = index;
        this.historyLength = historyLength;
    }

    @Override
    public List<Object> variableKeys() {
        return null;
    }

    @Override
    public Object get(Object variableKey) {
        return null;
    }

    @Override
    public State copy() {
        FrameHistoryState newState = new FrameHistoryState(this.index, this.historyLength);
        return newState;
    }
}