package edu.brown.cs.atari_vision.caffe.policies;

import burlap.behavior.policy.Policy;

/**
 * Created by maroderi on 7/13/16.
 */
public interface StatefulPolicy extends Policy {
    void loadStateAt(int steps);
}
