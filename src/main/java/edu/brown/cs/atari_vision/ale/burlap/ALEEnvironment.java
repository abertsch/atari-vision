package edu.brown.cs.atari_vision.ale.burlap;

import burlap.mdp.core.Action;
import burlap.mdp.core.Domain;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.EnvironmentOutcome;
import edu.brown.cs.atari_vision.ale.gui.AgentGUI;
import edu.brown.cs.atari_vision.ale.io.ALEDriver;
import edu.brown.cs.atari_vision.ale.io.Actions;
import edu.brown.cs.atari_vision.ale.io.RLData;
import edu.brown.cs.atari_vision.ale.movie.MovieGenerator;
import edu.brown.cs.atari_vision.ale.screen.NTSCPalette;
import edu.brown.cs.atari_vision.ale.screen.ScreenConverter;
import org.bytedeco.javacpp.opencv_core.*;

import java.io.IOException;

import static edu.brown.cs.atari_vision.ale.io.ALEDriver.PoolingType.POOLING_TYPE_MAX;

/**
 * @author Melrose Roderick
 */
public class ALEEnvironment<StateT extends State> implements Environment {

    MovieGenerator movieGenerator;

    /** The UI used for displaying images and receiving actions */
    private AgentGUI ui;
    /** The I/O object used to communicate with ALE */
    public ALEDriver io;

    protected ScreenConverter screenConverter;

    /** State data **/
    Domain domain;
    protected ALEStateGenerator<StateT> stateGenerator;
    protected StateT currentState;
    protected double lastReward;
    protected boolean isTerminal;

    /** Parameters */
    protected boolean useGUI;
    protected String rom;

    public ALEEnvironment(Domain domain, ALEStateGenerator stateGenerator, String rom, boolean useGUI) {
        this(domain, stateGenerator, rom, 1, useGUI);
    }

    public ALEEnvironment(Domain domain, ALEStateGenerator stateGenerator, String rom, int frameSkip, boolean useGUI) {
        this.rom = rom;
        this.useGUI = useGUI;
        if (this.useGUI) {
            // Create the GUI
            ui = new AgentGUI();
        }

        // Create the relevant I/O objects
        initIO(frameSkip);

        screenConverter = new ScreenConverter();

        // Set initial state
        this.stateGenerator = stateGenerator;
        this.currentState = this.stateGenerator.initialState(io.getScreen());
        this.domain = domain;
    }

    public void startRecording(String movieOutputFile) {
        movieGenerator = new MovieGenerator(movieOutputFile);
    }

    private void recordScreen(Mat screen) {
        // Save screen capture
        if (movieGenerator != null) {
            movieGenerator.record(screenConverter.convert(screen));
        }
    }

    @Override
    public StateT currentObservation() {
        return currentState;
    }

    @Override
    public EnvironmentOutcome executeAction(Action a) {
        // save start state
        StateT startState = currentState;

        // perform action
        int action = Actions.map(a.actionName());
        boolean closed = io.act(action);
        if (closed) {
            // the FIFO stream was closed
            // exit cleanly
            if (useGUI) {
                ui.die();
            }
            System.exit(0);
        }

        // Obtain the screen matrix
        Mat screen = io.getScreen();
        // Pass it on to UI
        if (useGUI) {
            ui.updateImage(screen);
        }

        // Get RLData
        RLData rlData = io.getRLData();

        // Update Environment State
        lastReward = rlData.reward;
        isTerminal = rlData.isTerminal;
        currentState = stateGenerator.nextState(screen, currentState, a, lastReward, isTerminal);

        // Record Screen for movie
        recordScreen(screen);

        return new EnvironmentOutcome(startState, a, currentState, lastReward, isInTerminalState());
    }

    @Override
    public double lastReward() {
        return lastReward;
    }

    @Override
    public boolean isInTerminalState() {
        return isTerminal;
    }

    @Override
    public void resetEnvironment() {
        // perform reset action
        io.act(Actions.map("system_reset"));
        isTerminal = false;

        // reset initialState
        currentState = stateGenerator.initialState(io.getScreen());
    }

    public void setStateGenerator(ALEStateGenerator<StateT> stateGenerator) {
        this.stateGenerator = stateGenerator;
    }

    public void die() {
        if (useGUI) {
            ui.die();
        }
    }

    /** Initialize the I/O object.
     *
     */
    protected void initIO(int frameSkip) {
        io = null;

        try {
            // Initialize the pipes; use named pipes if requested
            io = new ALEDriver(rom, frameSkip);

            // Determine which information to request from ALE
            io.setUpdateScreen(true);
            io.setUpdateRL(true);
            io.setUpdateRam(false);
            io.setPoolingType(POOLING_TYPE_MAX);
            io.initPipes();
        }
        catch (IOException e) {
            System.err.println ("Could not initialize pipes: "+e.getMessage());
            System.exit(-1);
        }
    }
}
