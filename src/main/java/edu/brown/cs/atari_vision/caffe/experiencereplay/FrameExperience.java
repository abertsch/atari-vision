package edu.brown.cs.atari_vision.caffe.experiencereplay;

import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;

import java.io.Serializable;

/**
 * Created by maroderi on 6/29/16.
 */
public class FrameExperience implements Serializable {

    /** The action id of the action that was taken */
    public int a;

    /** The reward received */
    public double r;

    /** True if the experience resulted in a terminal state */
    public boolean terminated;

    /** The state from which the action was taken */
    public FrameHistoryState o;

    /** The state to which the agent arrived at */
    public FrameHistoryState op;

    public FrameExperience(FrameHistoryState o, int a, FrameHistoryState op, double r, boolean terminated) {
        this.o = o;
        this.a = a;
        this.op = op;
        this.r = r;
        this.terminated = terminated;
    }
}
