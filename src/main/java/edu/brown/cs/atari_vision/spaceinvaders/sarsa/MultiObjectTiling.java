package edu.brown.cs.atari_vision.spaceinvaders.sarsa;

import burlap.behavior.functionapproximation.DifferentiableStateActionValue;
import burlap.behavior.functionapproximation.FunctionGradient;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.state.State;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by MelRod on 4/26/16.
 */
public class MultiObjectTiling implements DifferentiableStateActionValue, Serializable {

    public ArrayList<ObjectTiling> objectTilings;

    int numParams;

    public MultiObjectTiling(List<ObjectTiling> objectTilings) {
        this.objectTilings = new ArrayList<>(objectTilings);

        numParams = 0;
        for (ObjectTiling objectTiling : objectTilings) {
            numParams += objectTiling.numParameters();
        }
    }

    @Override
    public FunctionGradient gradient(State state, Action abstractGroundedAction) {
        FunctionGradient.SparseGradient fullGradient = new FunctionGradient.SparseGradient(numParams);

        int indexOffset = 0;
        for (ObjectTiling objectTiling : objectTilings) {
            FunctionGradient gradient = objectTiling.gradient(state, abstractGroundedAction);

            for (FunctionGradient.PartialDerivative pd : gradient.getNonZeroPartialDerivatives()) {
                fullGradient.put(indexOffset + pd.parameterId, pd.value);
            }

            indexOffset += objectTiling.numParameters();
        }

        return fullGradient;
    }

    @Override
    public double evaluate(State state, Action abstractGroundedAction) {
        int sum = 0;
        for (ObjectTiling objectTiling : objectTilings) {
            sum += objectTiling.evaluate(state, abstractGroundedAction);
        }
        return sum;
    }

    @Override
    public int numParameters() {
        return numParams;
    }

    @Override
    public double getParameter(int i) {
        for (ObjectTiling objectTiling : objectTilings) {
            if (i < objectTiling.numParameters()) {
                return objectTiling.getParameter(i);
            } else {
                i -= objectTiling.numParameters();
            }
        }
        throw new IndexOutOfBoundsException();
    }

    @Override
    public void setParameter(int i, double v) {
        for (ObjectTiling objectTiling : objectTilings) {
            if (i < objectTiling.numParameters()) {
                objectTiling.setParameter(i, v);
                return;
            } else {
                i -= objectTiling.numParameters();
            }
        }
        throw new IndexOutOfBoundsException();
    }

    @Override
    public void resetParameters() {
        for (ObjectTiling objectTiling : objectTilings) {
            objectTiling.resetParameters();
        }
    }

    @Override
    public ParametricStateActionFunction copy() {
        ArrayList<ObjectTiling> newObjectTilings = new ArrayList<>(objectTilings.size());

        for (int i = 0; i < objectTilings.size(); i++) {
            newObjectTilings.set(i, (ObjectTiling)objectTilings.get(i).copy());
        }

        return new MultiObjectTiling(newObjectTilings);
    }
}
