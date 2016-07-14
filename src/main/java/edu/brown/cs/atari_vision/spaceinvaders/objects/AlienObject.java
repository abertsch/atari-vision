package edu.brown.cs.atari_vision.spaceinvaders.objects;

import burlap.mdp.core.state.State;
import edu.brown.cs.atari_vision.spaceinvaders.SIDomainConstants;

/**
 * @author Melrose Roderick
 */
public class AlienObject extends SIObject {

    public AlienObject(int x, int y, AgentObject agent) {
        super(x, y, agent);
    }

    @Override
    public String className() {
        return SIDomainConstants.CLASSALIEN;
    }

    @Override
    public State copy() {
        return new AgentObject(this.x, this.y);
    }
}
