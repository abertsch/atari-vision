package edu.brown.cs.atari_vision.spaceinvaders.cv;

import edu.brown.cs.atari_vision.spaceinvaders.SIDomainConstants;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Created by alex on 14/4/16.
 */
public class TargetedContourFinder {

    // Scalars
    private final Scalar ALIEN_COLOR = new Scalar(29, 134, 134, 1);
    private final Scalar PLAYER_COLOR = new Scalar(50, 132, 50, 1);
    private final Scalar BOMB_COLOR = new Scalar(40, 83, 181, 1);

    // Shared targeted sprite finder
    private static TargetedContourFinder tcf = new TargetedContourFinder();
    public static TargetedContourFinder getTCF() {
        return tcf;
    }

    public TargetedContourFinder() {
    }

    /**
     * Finds center points for sprites within an image, searching for only a given color.
     * @param mat a binary image to find contours in.
     * @param color The color to search for in the image.
     * @param min The maximum size of a contour in the image.
     * @param max The maximimum size of a contour in the image.
     * @return A list of points, each centered on one fo the bounding boxes found in the image.
     */
    private ArrayList<Point> findSpritePoints(Mat mat, Scalar color, double min, double max) {
        // Convert image to binary
        Mat binary = new Mat();
        // TODO: convert this code
//        inRange(mat, color, color, binary);

        // Find contours
        ArrayList<Point> points = new ArrayList<>();
        MatVector contours = new MatVector();
        findContours(binary, contours, new Mat(), RETR_LIST, CHAIN_APPROX_TC89_KCOS);
        for (int i = 0; i < contours.size(); i++) {
            Mat contour = contours.get(i);

            final double area = contourArea(contour);
            if (area >= min && area <= max) {
                Rect rect = boundingRect(contour);
                points.add(new Point((int)(rect.x() + 0.5 * rect.width()),
                        (int)(rect.y() + 0.5 * rect.height())));
            }
        }
        return points;
    }

    public HashMap<String, List<Point>> findSprites(Mat screen) {
        // List to contain all lists of sprite locations
        HashMap<String, List<Point>> locations = new HashMap<>();

        // Filter to select only aliens
        ArrayList<Point> aliens = findSpritePoints(screen, ALIEN_COLOR, 8, 100);
        locations.put(SIDomainConstants.CLASSALIEN, aliens);

        // Filter to select only player
        ArrayList<Point> player = findSpritePoints(screen, PLAYER_COLOR, 5, 50);
        locations.put(SIDomainConstants.CLASSAGENT, player);

        // Filter to select only bombs
        ArrayList<Point> bombs = findSpritePoints(screen, BOMB_COLOR, 0, 5);
        locations.put(SIDomainConstants.CLASS_BOMB_UNKNOWN, bombs);

        return locations;
    }
}
