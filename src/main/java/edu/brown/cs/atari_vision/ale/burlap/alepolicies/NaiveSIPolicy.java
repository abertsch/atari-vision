package edu.brown.cs.atari_vision.ale.burlap.alepolicies;

import burlap.behavior.policy.support.ActionProb;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.SimpleAction;
import burlap.mdp.core.oo.propositional.GroundedProp;
import burlap.mdp.core.oo.propositional.PropositionalFunction;
import burlap.mdp.core.oo.state.OOState;
import burlap.mdp.core.oo.state.ObjectInstance;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.oo.OOSADomain;
import edu.brown.cs.atari_vision.ale.burlap.ALEDomainConstants;
import burlap.behavior.policy.Policy;

import java.util.List;

/**
 * Created by MelRod on 3/22/16.
 */
public class NaiveSIPolicy implements Policy {

    /**
     * The actions from which selection is performed
     */
    private Action rightAction;
    private Action rightFireAction;
    private Action leftAction;
    private Action leftFireAction;

    private PropositionalFunction vertAlignPF;

    private boolean movingRight;

    public NaiveSIPolicy(OOSADomain domain) {
        movingRight = true;

        rightAction = new SimpleAction("player_a_right");
        rightFireAction = new SimpleAction("player_a_rightfire");
        leftAction = new SimpleAction("player_a_left");
        leftFireAction = new SimpleAction("player_a_leftfire");

        vertAlignPF = domain.propFunction(ALEDomainConstants.PFVertAlign);
    }

    @Override
    public Action action(State s) {

        OOState state = (OOState)s;

        List <ObjectInstance> instances = state.objectsOfClass(ALEDomainConstants.CLASSAGENT);

        if (instances.size() > 0) {
            ObjectInstance agent = instances.get(0);

            int x = (Integer)agent.get(ALEDomainConstants.XATTNAME);

            if (movingRight && x >= 115) {
                movingRight = false;
            } else if (!movingRight && x <= 40) {
                movingRight = true;
            }
        }

        // check whether or not there is an alien overhead
        boolean hasVertAlign = false;
        List<GroundedProp> gps = vertAlignPF.allGroundings(state);
        for(GroundedProp gp : gps) {
            if (gp.isTrue(state)) {
                hasVertAlign = true;
                break;
            }
        }

        // execute naive action
        if (hasVertAlign) {
            if (movingRight) {
                return rightFireAction;
            } else {
                return leftFireAction;
            }
        } else {
            if (movingRight) {
                return rightAction;
            } else {
                return leftAction;
            }
        }
    }

    @Override
    public double actionProb(State s, Action a) {
        return 0;
    }

    @Override
    public boolean definedFor(State s) {
        return true;
    }
}
