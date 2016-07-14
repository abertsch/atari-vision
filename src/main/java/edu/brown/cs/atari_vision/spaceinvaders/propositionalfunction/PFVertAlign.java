package edu.brown.cs.atari_vision.spaceinvaders.propositionalfunction;

import burlap.mdp.core.oo.state.OOState;
import burlap.mdp.core.oo.state.ObjectInstance;
import burlap.mdp.core.oo.propositional.PropositionalFunction;
import edu.brown.cs.atari_vision.spaceinvaders.SIDomainConstants;

/**
 * Created by MelRod on 3/12/16.
 */
public class PFVertAlign extends PropositionalFunction {

    int width;

    public PFVertAlign(String name, String[] parameterClasses, int width) {
        super(name, parameterClasses);

        this.width = width;
    }

    @Override
    public boolean isTrue(OOState s, String... params) {

        ObjectInstance objectA = s.object(params[0]);
        ObjectInstance objectB = s.object(params[1]);

        int xA = (Integer)objectA.get(SIDomainConstants.XATTNAME);
        int xB = (Integer)objectB.get(SIDomainConstants.XATTNAME);
        if (width/2 >= Math.abs(xA - xB)) {
            return true;
        } else {
            return false;
        }
    }
}
