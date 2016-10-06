package edu.brown.cs.atari_vision.spaceinvaders;

import burlap.mdp.auxiliary.DomainGenerator;
import burlap.mdp.core.action.ActionType;
import burlap.mdp.core.action.SimpleAction;
import burlap.mdp.core.oo.OODomain;
import burlap.mdp.core.oo.propositional.PropositionalFunction;
import burlap.mdp.core.oo.state.OOState;
import burlap.mdp.core.oo.state.ObjectInstance;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.visualizer.*;
import edu.brown.cs.atari_vision.ale.burlap.ALEDomainConstants;
import edu.brown.cs.atari_vision.ale.burlap.ALEDomainGenerator;
import edu.brown.cs.atari_vision.ale.burlap.action.ActionSet;
import edu.brown.cs.atari_vision.ale.burlap.propositionalfunction.PFVertAlign;
import edu.brown.cs.atari_vision.ale.io.Actions;
import edu.brown.cs.atari_vision.spaceinvaders.cv.SpriteFinder;
import edu.brown.cs.atari_vision.spaceinvaders.sarsa.MultiObjectTiling;
import edu.brown.cs.atari_vision.spaceinvaders.sarsa.ObjectTiling;

import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.List;

/**
 * Space Invaders Domain Generator
 *
 * Created by MelRod on 3/12/16.
 */
public class SIDomainGenerator implements DomainGenerator {

    public static int objectWidth = 8;

    SpriteFinder spriteFinder;
    ActionSet actionSet;

    public SIDomainGenerator() {
        actionSet = Actions.siActionSet();
    }

    @Override
    public OOSADomain generateDomain() {

        // adds all SI Actions
        OOSADomain domain = new OOSADomain();
        for (ActionType actionType : actionSet.actionTypeList()) {
            domain.addActionType(actionType);
        }

        // initialize SpriteFinder
        spriteFinder = new SpriteFinder();


        // Propositional Functions
        OODomain.Helper.addPfsToDomain(domain, generatePfs());


        return domain;
    }

    public List<PropositionalFunction> generatePfs(){
        List<PropositionalFunction> pfs = new ArrayList<>();

        pfs.add(new PFVertAlign(ALEDomainConstants.PFVertAlign,
                new String[]{ALEDomainConstants.CLASSAGENT, ALEDomainConstants.CLASSALIEN},
                objectWidth));
        new PFVertAlign(
                ALEDomainConstants.PFVertAlign,
                new String[]{ALEDomainConstants.CLASSAGENT, ALEDomainConstants.CLASS_BOMB_ALIEN},
                objectWidth);

        return pfs;
    }

    public StateRenderLayer getStateRenderLayer(){
        StateRenderLayer rl = new StateRenderLayer();


        //create tilings
        int numTilesX = 14;
        int numTilesY = 20;
        ArrayList<ObjectTiling> tilings = new ArrayList<>();


        // Add Agent tiling
        tilings.add(new ObjectTiling(
                ALEDomainConstants.CLASSAGENT,
                ALEDomainConstants.XATTNAME,
                ALEDomainConstants.YATTNAME,
                0, 0, ALEDomainConstants.ALEScreenWidth, ALEDomainConstants.ALEScreenHeight,
                numTilesX, 1
        ));

        // Add Alien tiling
        tilings.add(new ObjectTiling(
                ALEDomainConstants.CLASSALIEN,
                ALEDomainConstants.AGENT_CENT_XATTNAME,
                ALEDomainConstants.AGENT_CENT_YATTNAME,
                -ALEDomainConstants.ALEScreenWidth, -ALEDomainConstants.ALEScreenHeight,
                ALEDomainConstants.ALEScreenWidth, ALEDomainConstants.ALEScreenHeight,
                2*numTilesX, 2*numTilesY
        ));

        // Add Bomb tiling
        tilings.add(new ObjectTiling(
                ALEDomainConstants.CLASS_BOMB_ALIEN,
                ALEDomainConstants.AGENT_CENT_XATTNAME,
                ALEDomainConstants.AGENT_CENT_YATTNAME,
                -ALEDomainConstants.ALEScreenWidth,  -ALEDomainConstants.ALEScreenHeight,
                ALEDomainConstants.ALEScreenWidth, ALEDomainConstants.ALEScreenHeight,
                2*numTilesX, 2*numTilesY
        ));


        rl.addStatePainter(new TilingPainter(new MultiObjectTiling(tilings)));

        OOStatePainter statePainter = new OOStatePainter();
        statePainter.addObjectClassPainter(ALEDomainConstants.CLASSAGENT, new SIPainter(Color.GREEN));
        statePainter.addObjectClassPainter(ALEDomainConstants.CLASSALIEN, new SIPainter(Color.YELLOW));
        statePainter.addObjectClassPainter(ALEDomainConstants.CLASS_BOMB_AGENT, new SIPainter(Color.ORANGE));
        statePainter.addObjectClassPainter(ALEDomainConstants.CLASS_BOMB_ALIEN, new SIPainter(Color.RED));
        statePainter.addObjectClassPainter(ALEDomainConstants.CLASS_BOMB_UNKNOWN, new SIPainter(Color.GRAY));
        rl.addStatePainter(statePainter);

        return rl;
    }

    public Visualizer getVisualizer(){
        return new Visualizer(this.getStateRenderLayer());
    }

    public class SIPainter implements ObjectPainter {

        Color color;

        public SIPainter(Color color) {
            super();

            this.color = color;
        }

        @Override
        public void paintObject(Graphics2D g2, OOState s, ObjectInstance ob, float cWidth, float cHeight) {

            //agent will be filled in gray
            g2.setColor(color);

            //set up floats for the width and height of our domain
            float fWidth = ALEDomainConstants.ALEScreenWidth;
            float fHeight = ALEDomainConstants.ALEScreenHeight;

            //determine the width of a single atari pixel on our canvas
            //such that the whole map can be painted
            float pixelWidth = (cWidth / fWidth);
            float pixelHeight = (cHeight / fHeight);

            float ax = (Integer)ob.get(ALEDomainConstants.XATTNAME);
            float ay = fHeight - (Integer)ob.get(ALEDomainConstants.YATTNAME);

            //left coordinate of cell on our canvas
            float rx = ax*pixelWidth;

            //top coordinate of cell on our canvas
            //coordinate system adjustment because the java canvas
            //origin is in the top left instead of the bottom right
            float ry = cHeight - ay*pixelHeight;

            //paint the rectangle
            g2.fill(new Ellipse2D.Float(rx - objectWidth/2, ry - objectWidth/2, objectWidth, objectWidth));
        }
    }

    public class TilingPainter implements StatePainter{

        MultiObjectTiling multiTiling;

        public TilingPainter(MultiObjectTiling multiTiling) {
            this.multiTiling = multiTiling;
        }

        @Override
        public void paint(Graphics2D g2, State s, float cWidth, float cHeight) {

            OOState state = (OOState)s;

            // get agent coordinates
            java.util.List<ObjectInstance> agentInstances = state.objectsOfClass(ALEDomainConstants.CLASSAGENT);
            float agentX;
            float agentY;
            if (agentInstances.size() > 0) {
                ObjectInstance agentInstance = agentInstances.get(0);
                agentX = ((Integer)agentInstance.get(ALEDomainConstants.XATTNAME)/(float)ALEDomainConstants.ALEScreenWidth)*cWidth;
                agentY = ((Integer)agentInstance.get(ALEDomainConstants.YATTNAME)/(float)ALEDomainConstants.ALEScreenHeight)*cHeight;
            } else {
                agentX = 0;
                agentY = 0;
            }

            // update tiling for state
            multiTiling.evaluate(state, new SimpleAction(Actions.actionNames[0]));

//            // make grid
//            g2.setColor(Color.black);
//            g2.setStroke(new BasicStroke(2));
//            for (int x = 1; x < tiling.numTilesX; x++) {
//                g2.drawLine((int)(x*tileWidth), 0, (int)(x*tileWidth), (int)cHeight);
//            }
//            for (int y = 1; y < tiling.numTilesY; y++) {
//                g2.drawLine(0, (int)(y*tileHeight), (int)cWidth, (int)(y*tileHeight));
//            }

            for (ObjectTiling tiling : multiTiling.objectTilings) {

                if (tiling.objectClass.equals(ALEDomainConstants.CLASSAGENT)) {
                    g2.setColor(Color.cyan);
                } else if (tiling.objectClass.equals(ALEDomainConstants.CLASSALIEN)) {
                    g2.setColor(Color.red);
                } else if (tiling.objectClass.equals(ALEDomainConstants.CLASS_BOMB_ALIEN)) {
                    g2.setColor(Color.black);
                } else {
                    g2.setColor(Color.black);
                }

                // tile sizes
                float tileWidth;
                float tileHeight;
                if (tiling.xAttributeName.equals(ALEDomainConstants.AGENT_CENT_XATTNAME)) {
                    tileWidth = cWidth/(tiling.numTilesX/2);
                    tileHeight = cHeight/(tiling.numTilesY/2);
                } else {
                    tileWidth = cWidth/tiling.numTilesX;
                    tileHeight = cHeight/tiling.numTilesY;
                }

                // make fill
                for (int i = 0; i < tiling.numTiles; i++) {
                    if (tiling.bitSet.get(i)) {
                        int x = (i % tiling.numTilesX);
                        int y = (i / tiling.numTilesX);

                        if (tiling.xAttributeName.equals(ALEDomainConstants.AGENT_CENT_XATTNAME)) {
                            g2.fill(new Rectangle2D.Float(x * tileWidth - cWidth + agentX, y * tileHeight - cHeight + agentY, tileWidth, tileHeight));
                        } else {
                            g2.fill(new Rectangle2D.Float(x * tileWidth, y * tileHeight, tileWidth, tileHeight));
                        }
                    }
                }

//                if (tiling.objectClass.equals(ALEDomainConstants.CLASSALIEN)) {
//                    // print grid
//                    System.err.println();
//                    for (int i = 0; i < tiling.numTiles; i++) {
//                        if (i % tiling.numTilesX == 0) {
//                            System.err.println();
//                        }
//
//                        if (tiling.bitSet.get(i)) {
//                            System.err.print("1 ");
//                        } else {
//                            System.err.print("0 ");
//                        }
//                    }
//                    System.err.println();
//                }
            }
        }
    }


    public Visualizer getParamVisualizer(ObjectTiling tiling, int actionID){
        StateRenderLayer rl = new StateRenderLayer();

        rl.addStatePainter(new TilingParamPainter(tiling, actionID));

        return new Visualizer(rl);
    }
    public class TilingParamPainter implements StatePainter{

        ObjectTiling tiling;
        int actionID;

        public TilingParamPainter(ObjectTiling tiling, int actionID) {
            this.tiling = tiling;
            this.actionID = actionID;
        }

        @Override
        public void paint(Graphics2D g2, State state, float cWidth, float cHeight) {

            int indexOffset = actionID * tiling.numTiles;

            // tile sizes
            float tileWidth = cWidth/tiling.numTilesX;
            float tileHeight = cHeight/tiling.numTilesY;

            double maxParam = 0;

            // find max
            for (int i = 0; i < tiling.numTiles; i++) {

                double param = tiling.params[i + indexOffset];
                if (param > maxParam) {
                    maxParam = param;
                } else if (-param > maxParam) {
                    maxParam = -param;
                }
            }
//            System.err.println("MAX PARAM: " + maxParam);

            // make fill
            for (int i = 0; i < tiling.numTiles; i++) {

                double param = tiling.params[i + indexOffset];
                int colorVal = (int)((Math.abs(param)/maxParam)*255.0);

                Color color;
                if (param > 0) {
                    color = new Color(0, colorVal, 0);
                } else {
                    color = new Color(colorVal, 0, 0);
                }
                g2.setColor(color);


                int x = (i % tiling.numTilesX);
                int y = (i / tiling.numTilesX);

                g2.fill(new Rectangle2D.Float(x * tileWidth, y * tileHeight, tileWidth, tileHeight));
            }

            // draw agent
            g2.setColor(Color.cyan);
            g2.fill(new Ellipse2D.Float(cWidth/2-5, cHeight/2-5, 10, 10));
        }
    }
}
