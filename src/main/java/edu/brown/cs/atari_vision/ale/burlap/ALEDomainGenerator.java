package edu.brown.cs.atari_vision.ale.burlap;

import burlap.mdp.auxiliary.DomainGenerator;
import burlap.mdp.core.action.SimpleAction;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.core.action.ActionType;
import burlap.mdp.singleagent.oo.OOSADomain;
import edu.brown.cs.atari_vision.ale.burlap.action.ActionSet;
import edu.brown.cs.atari_vision.ale.burlap.action.SimpleActionType;
import edu.brown.cs.atari_vision.ale.io.Actions;

/**
 * Created by MelRod on 3/18/16.
 */
public class ALEDomainGenerator implements DomainGenerator {

    ActionSet actionSet;

    public ALEDomainGenerator() {
        super();
        actionSet = Actions.saActionSet();
    }

    public ALEDomainGenerator(ActionSet actionSet) {
        super();

        this.actionSet = actionSet;
    }

    @Override
    public SADomain generateDomain() {
        SADomain domain = new SADomain();

        // add in NullActions for Domain
        for (ActionType actionType : actionSet.actionTypeList()) {
            domain.addActionType(actionType);
        }

        return domain;
    }
}
