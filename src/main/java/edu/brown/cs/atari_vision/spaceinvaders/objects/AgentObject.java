package edu.brown.cs.atari_vision.spaceinvaders.objects;

import burlap.mdp.core.state.State;
import edu.brown.cs.atari_vision.spaceinvaders.SIDomainConstants;

/**
 * @author Melrose Roderick
 */
public class AgentObject extends SIObject {

    public AgentObject(int x, int y) {
        super(x, y, 0, 0);
    }

    @Override
    public String className() {
        return SIDomainConstants.CLASSAGENT;
    }

    @Override
    public State copy() {
        return new AgentObject(this.x, this.y);
    }
}
