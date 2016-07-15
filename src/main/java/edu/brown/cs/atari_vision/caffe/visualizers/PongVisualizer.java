package edu.brown.cs.atari_vision.caffe.visualizers;

import burlap.behavior.valuefunction.QValue;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.presets.caffe;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by maroderi on 6/17/16.
 */
public class PongVisualizer extends JPanel {

    public static List<QValue> qValues;

    public static final int NUM_ACTIONS = 4;
    public static final float MAX_Q_VALUE = 1;

    public static final int DEFAULT_HEIGHT = 300;
    public static final int DEFAULT_BAR_WIDTH = 300;

    public static final Color[] COLORS = new Color[]{Color.CYAN, Color.GREEN, Color.MAGENTA, Color.PINK};

    public PongVisualizer() {

        //define defaults for the JFrame
        setBackground(Color.WHITE);

        setPreferredSize(new Dimension(DEFAULT_BAR_WIDTH * NUM_ACTIONS, DEFAULT_HEIGHT));

        qValues = new ArrayList<>();
        for (int a = 0; a < NUM_ACTIONS; a++) {
            qValues.add(new QValue(null, null, 0));
        }

        ActionListener al = new ActionListener() {
            public void actionPerformed(ActionEvent ae) {
                //refresh screen
                repaint();
            }
        };

        Timer t = new Timer(33, al);
        t.start();
    }

    @Override
    public void paintComponent(Graphics g) {

        super.paintComponent(g);

        int height = this.getHeight();
        float semiHeight = height/2.0f;
        int width = this.getWidth();

        int barWidth = width/NUM_ACTIONS;

        g.setFont(new Font("Arial", Font.PLAIN, 20));

//        double norm = 1;
        double norm = 1e-10;
        double absMax = 0;
        for (int a = 0; a < NUM_ACTIONS; a++) {
            double absQ = Math.abs(qValues.get(a).q);

            if (absQ > absMax) {
                absMax = absQ;
            }
        }
        while (norm < 1e10) {
            if (norm > absMax) {
                break;
            }
            norm *= 10;
        }

        for (int a = 0; a < NUM_ACTIONS; a++) {
            double q = qValues.get(a).q;

            int x = a * barWidth;

            int y;
            int barHeight;
            if (q > 0) {
                y = (int)(semiHeight - (q/norm)*semiHeight);
                barHeight = (int) (semiHeight - y);
            } else {
                y = (int) semiHeight;
                barHeight = (int) (-(q/norm)*semiHeight);
            }

            g.setColor(COLORS[a]);
            g.fillRect(x, y, barWidth, barHeight);
        }

        int textHeight = 15;
        int paddingSize = 5;

        g.setColor(Color.black);
        g.drawString(String.format("%.3f", norm), 0, textHeight + paddingSize);
        g.drawString(String.format("%.3f", -norm), 0, height - paddingSize);

        int axisHeight = 4;
        g.setColor(Color.black);
        g.fillRect(0, (int)(semiHeight - axisHeight/2.0), width, axisHeight);
    }

    public static void setQValues(List<QValue> newQValues) {
        qValues = new ArrayList<>(newQValues);
    }

    public static JFrame createPongVisualizer() {
        JFrame f = new JFrame("Pong Visualizer");
        f.setDefaultCloseOperation( JFrame.EXIT_ON_CLOSE );
        f.add(new PongVisualizer());
        f.pack();
        f.setLocationByPlatform(true);
        f.setResizable( true );
        f.setVisible( true );

        return f;
    }

    public static void main(String[] args) {

        Loader.load(caffe.class);

        // Test
        createPongVisualizer();

        List<QValue> qValues = new ArrayList<>();
        qValues.add(new QValue(null, null, -0.1f));
        qValues.add(new QValue(null, null, 0.2f));
        qValues.add(new QValue(null, null, -0.8f));

        setQValues(qValues);
    }
}
