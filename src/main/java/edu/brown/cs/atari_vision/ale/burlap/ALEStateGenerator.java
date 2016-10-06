package edu.brown.cs.atari_vision.ale.burlap;

import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;

import static org.bytedeco.javacpp.opencv_core.*;

/**
 * Created by maroderi on 6/22/16.
 */
public interface ALEStateGenerator<StateT extends State> {

    StateT initialState(Mat screen);

    StateT nextState(Mat screen, StateT prevState, Action action, double reward, boolean terminated);
}
