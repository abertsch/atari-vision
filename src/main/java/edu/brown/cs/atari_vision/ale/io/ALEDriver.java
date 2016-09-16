package edu.brown.cs.atari_vision.ale.io;

import edu.brown.cs.atari_vision.ale.burlap.ALEDomainConstants;
import edu.brown.cs.atari_vision.ale.screen.ColorPalette;
import edu.brown.cs.atari_vision.ale.screen.NTSCPalette;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.opencv_core.*;

import java.awt.*;
import java.io.*;
import java.nio.ByteBuffer;

import static org.bytedeco.javacpp.opencv_core.*;

/**
 * Created by MelRod on 5/23/16.
 */
public class ALEDriver {

    static final String ALE_FILE = "./ale";
    static final String ROM_DIR = "roms/";
    static final String ALE_ERROR_FILE = "ale_err.txt";

    protected Process process;

    /** Data structure holding the screen image */
    Mat screen;
    /** Data structure holding the raw frame data. If pooling != 0, pool over these frames **/
    Mat frameA;
    Mat frameB;
    PoolingType poolingType;
    public enum PoolingType {
        POOLING_TYPE_NONE,
        POOLING_TYPE_MAX,
        POOLING_TYPE_MEAN
    }
    /** Data structure holding colors */
    ColorPalette colorPalette = new NTSCPalette();
    /** Data structure holding the RAM data */
    protected ConsoleRAM ram;
    /** Data structure holding RL data */
    protected RLData rlData;
    /** Whether termination was requested from the I/O channel */
    protected boolean terminateRequested;

    /** Input object */
    protected final BufferedReader in;
    /** Output object */
    protected final PrintStream out;

    /** Flags indicating the kind of data we want to receive from ALE */
    protected boolean updateScreen, updateRam, updateRLData;
    /** We will request that ALE sends data every 'frameskip' frames. */
    protected int frameskip;

    /** The action we send for player B (always noop in this case) */
    protected final int playerBAction = Actions.map("player_b_noop");

    /** A state variable used to track of whether we should receive or send data */
    protected boolean hasObserved;

    protected boolean useRLE = true;

    public ALEDriver(String rom) {
        this(rom, 1);
    }

    public ALEDriver(String rom, int frameskip) {
        this.frameskip = frameskip;
        startALE(rom);

        in = new BufferedReader(new InputStreamReader(process.getInputStream()));
        out = new PrintStream(new BufferedOutputStream(process.getOutputStream()));
    }

    private void startALE(String rom) {
        ProcessBuilder pb = new ProcessBuilder(
                ALE_FILE,
                "-game_controller", "fifo",
                "-frame_skip", "0",
                "-repeat_action_probability", "0",
                "-disable_color_averaging", "true",
                (new File(ROM_DIR, rom)).getPath())
                .redirectError(new File(ALE_ERROR_FILE));

        try {
            process = pb.start();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /** Closes the I/O channel.
     *
     */
    public void close() {
        process.destroy();
    }

    public void setUpdateScreen(boolean updateScreen) {
        this.updateScreen = updateScreen;
    }

    public void setUpdateRam(boolean updateRam) {
        this.updateRam = updateRam;
    }

    public void setUpdateRL(boolean updateRL) {
        this.updateRLData = updateRL;
    }

    public void setPoolingType(PoolingType poolingType) {
        this.poolingType = poolingType;
    }

    /** A blocking method that sends initial information to ALE. See the
     *   documentation for protocol details.
     *
     */
    public void initPipes() throws IOException {
        // Read in the width and height of the screen
        // Format: <width>-<height>\n
        String line = in.readLine();
        String[] tokens = line.split("-");
        int width = Integer.parseInt(tokens[0]);
        int height = Integer.parseInt(tokens[1]);

        // Do some error checking - our width and height should be positive
        if (width <= 0 || height <= 0) {
            throw new RuntimeException("Invalid width/height: "+width+"x"+height);
        }

        // Create the data structures used to store received information
        screen = new Mat(height, width, CV_8UC3);
        frameA = new Mat(height, width, CV_8UC3);
        frameB = new Mat(height, width, CV_8UC3);
        ram = new ConsoleRAM();
        rlData = new RLData();

        // Now send back our preferences
        // Format: <wants-screen>,<wants-ram>,<frame-skip>,<wants-rldata>\n
        out.printf("%d,%d,%d,%d\n", updateScreen? 1:0, updateRam? 1:0, 1, updateRLData? 1:0);
        out.flush();

        // Initial observe
        observe(null);
    }

    public int getFrameSkip() {
        return frameskip;
    }

    /** Returns the screen matrix from ALE.
     *
     * @return
     */
    public Mat getScreen() {
        return screen;
    }

    /** Returns the RAM from ALE.
     *
     * @return
     */
    public ConsoleRAM getRAM() {
        return ram;
    }

    public RLData getRLData() {
        return rlData;
    }

    public boolean wantsTerminate() {
        return terminateRequested;
    }

    /** A blocking method which will get the next time step from ALE.
     *
     */
    public boolean observe(Mat outputFrame) {
        // Ensure that observe() is not called twice, as it will otherwise block
        //  as both ALE and the agent wait for data.
        if (hasObserved) {
            throw new RuntimeException("observe() called without subsequent act().");
        }
        else
            hasObserved = true;

        String line = null;

        // First read in a new line from ALE
        try {
            line = in.readLine();
            if (line == null) return true;
        }
        catch (IOException e) {
            return true;
        }

        // Catch the special keyword 'DIE'
        if (line.equals("DIE")) {
            terminateRequested = true;
            return false;
        }

        // Ignore blank lines (still send an action)
        if (line.length() > 0) {
            // The data format is:
            // <ram-string>:<screen-string>:<rl-data-string>:\n
            //  Some of these elements may be missing, in which case the separating
            //  colons are not sent. For example, if we only want ram and rl data,
            //  the format is <ram>:<rl-data>:

            String[] tokens = line.split(":");

            int tokenIndex = 0;

            // If necessary, first read the RAM data
            if (updateRam) {
                readRam(tokens[tokenIndex++]);
            }

            // Then update the screen
            if (updateScreen) {
                String screenString = tokens[tokenIndex++];

                if (outputFrame != null) {
                    if (useRLE)
                        readScreenRLE(screenString, outputFrame);
                    else
                        readScreenMatrix(screenString, outputFrame);
                }
            }

            // Finally obtain RL data
            if (updateRLData) {
                readRLData(tokens[tokenIndex++]);
            }
        }

        return false;
    }

    /** After a call to observe(), send back the necessary action.
     *
     * @param act
     * @return
     */
    public boolean act(int act) {
        // Ensure that we called observe() last
        if (!hasObserved) {
            throw new RuntimeException("act() called before observe().");
        }
        else
            hasObserved = false;

        boolean err = false;
        if (frameskip <= 1) {
            // Swap frameA and B
            Mat frameC = frameA;
            frameA = frameB;
            frameB = frameC;

            sendAction(act);
            err |= observe(frameA);
        } else {
            RLData rlData = new RLData();
            for (int f = 0; f < frameskip - 2; f++) {
                err |= actHelper(act, null, rlData);
            }
            err |= actHelper(act, frameB, rlData);
            err |= actHelper(act, frameA, rlData);

            this.rlData = rlData;
        }

        poolFrames();
        return err;
    }

    private boolean actHelper(int act, Mat frame, RLData rlData) {
        boolean err;

        sendAction(act); hasObserved = false;
        err = observe(frame);
        rlData.reward += this.rlData.reward;
        rlData.isTerminal |= this.rlData.isTerminal;
        rlData.lives = this.rlData.lives;

        return err;
    }

    protected void poolFrames() {
        if (frameB == null) {
            screen = frameA;
            return;
        }
        switch (poolingType) {
            case POOLING_TYPE_NONE:
                screen = frameA;
                break;
            case POOLING_TYPE_MAX:
                max(frameA, frameB, screen);
                break;
            case POOLING_TYPE_MEAN:
                addWeighted(frameA, 0.5, frameB, 0.5, 0, screen);
                break;
        }
    }

    /** Helper function to send out an action to ALE */
    public void sendAction(int act) {
        // Send player A's action, as well as the NOOP for player B
        // Format: <player_a_action>,<player_b_action>\n
        out.printf("%d,%d\n", act, playerBAction);
        out.flush();
    }

    /** Read in RL data from a given line.
     *
     * @param line
     */
    public void readRLData(String line) {
        // Parse RL data
        // Format: <is-terminal>:<reward>\n
        String[] tokens = line.split(",");

        // Parse the terminal bit
        rlData.isTerminal = (Integer.parseInt(tokens[0]) == 1);
        rlData.reward = Integer.parseInt(tokens[1]);
        rlData.lives = Integer.parseInt(tokens[2]);
    }

    /** Reads the console RAM from a string
     * @param line The RAM-part of the string sent by ALE.
     */
    public void readRam(String line) {
        int offset = 0;

        // Read in all of the RAM
        // Format: <r0><r1><r2>...<r127>
        //  where ri is 2 characters representing an integer between 0 and 0xFF
        for (int ptr = 0; ptr < ConsoleRAM.RAM_SIZE; ptr++) {
            int v = Integer.parseInt(line.substring(offset, offset + 2), 16);
            ram.ram[ptr] = v;

            offset += 2;
        }
    }

    /** Reads the screen matrix update from a string. The string only contains the
     *   pixels that differ from the previous frame.
     *
     * @param line The screen part of the string sent by ALE.
     */
    public void readScreenMatrix(String line, Mat frame) {
        BytePointer screenData = frame.data();
        int position = 0;

        byte[] buffer = new byte[frame.rows() * frame.cols() * frame.channels()];

        int ptr = 0;

        // 0.3 protocol - send everything
        for (int y = 0; y < frame.rows(); y++)
            for (int x = 0; x < frame.cols(); x++) {
                int v = byteAt(line, ptr);

                Color c = colorPalette.get(v);
                buffer[position] = (byte)c.getBlue();
                buffer[position + 1] = (byte)c.getGreen();
                buffer[position + 2] = (byte)c.getRed();

                position += 3;

                ptr += 2;
            }

        screenData.put(buffer);
    }

    /** Parses a hex byte in the given String, at position 'ptr'. */
    private int byteAt(String line, int ptr) {
        int ld = line.charAt(ptr+1);
        int hd = line.charAt(ptr);

        if (ld >= 'A') ld -= 'A' - 10;
        else ld -= '0';
        if (hd >= 'A') hd -= 'A' - 10;
        else hd -= '0';

        return (hd << 4) + ld;
    }

    /** Read in a run-length encoded screen. ALE 0.3-0.4 */
    public void readScreenRLE(String line, Mat frame) {

        BytePointer screenData = frame.data();
        int position = 0;

        byte[] buffer = new byte[frame.rows() * frame.cols() * frame.channels()];

        int ptr = 0;

        while (ptr < line.length()) {
            // Read in the next run
            int v = byteAt(line, ptr);
            int l = byteAt(line, ptr + 2);
            ptr += 4;

            Color c = colorPalette.get(v);
            for (int i = 0; i < l; i++) {
                buffer[position] = (byte)c.getBlue();
                buffer[position + 1] = (byte)c.getGreen();
                buffer[position + 2] = (byte)c.getRed();

                position += 3;
            }
        }

        screenData.put(buffer);
    }
}
