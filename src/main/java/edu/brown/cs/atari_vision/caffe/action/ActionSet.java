package edu.brown.cs.atari_vision.caffe.action;


import burlap.mdp.core.action.Action;
import burlap.mdp.core.action.ActionType;
import burlap.mdp.core.action.SimpleAction;
import burlap.mdp.core.action.UniversalActionType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by MelRod on 5/28/16.
 */
public class ActionSet {

    protected Action[] actions;
    protected Map<Action, Integer> actionMap;
    protected int size;

    public ActionSet(String[] actionNames) {
        size = actionNames.length;
        actions = new Action[size];
        for (int i = 0; i < size; i++) {
            actions[i] = new SimpleAction(actionNames[i]);
        }

        initActionMap();
    }

    public ActionSet(Action[] actions) {
        this.actions = actions;
        size = actions.length;

        initActionMap();
    }

    protected void initActionMap() {
        actionMap = new HashMap<>();
        for (int i = 0; i < actions.length; i++) {
            actionMap.put(actions[i], i);
        }
    }

    public Action get(int i) {
        return actions[i];
    }

    public int map(Action action) {
        return actionMap.get(action);
    }

    public int size() {
        return size;
    }
}