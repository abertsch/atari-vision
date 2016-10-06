package edu.brown.cs.atari_vision.ale.burlap.action;

import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.ActionType;
import burlap.mdp.core.action.SimpleAction;
import burlap.mdp.core.state.State;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by MelRod on 5/28/16.
 */
public class ActionSet {

    private String[] actions;
    private Map<String, Integer> actionMap;
    int size;

    public ActionSet(String[] actions) {
        this.actions = actions;
        size = actions.length;

        actionMap = new HashMap<>();
        for (int i = 0; i < actions.length; i++) {
            actionMap.put(actions[i], i);
        }
    }

    public String get(int i) {
        return actions[i];
    }

    public Action getAction(int i) {
        return new SimpleAction(actions[i]);
    }

    public int map(String action) {
        return actionMap.get(action);
    }

    public int size() {
        return size;
    }

    public List<Action> actionList() {
        List<Action> actionList = new ArrayList<>(actions.length);

        for (int i = 0; i < actions.length; i++) {
            Action a = new SimpleAction(actions[i]);
            actionList.add(a);
        }

        return actionList;
    }

    public List<ActionType> actionTypeList() {
        List<ActionType> actionList = new ArrayList<>(actions.length);

        for (int i = 0; i < actions.length; i++) {
            ActionType at = new SimpleActionType(actions[i], this);
            actionList.add(at);
        }

        return actionList;
    }
}
