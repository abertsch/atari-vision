package edu.brown.cs.atari_vision.caffe;

import edu.brown.cs.burlap.screen.ScreenConverter;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.caffe;
import org.bytedeco.javacpp.opencv_core;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

import static org.bytedeco.javacpp.opencv_core.CV_32F;
import static org.bytedeco.javacpp.opencv_core.CV_8U;
import static org.bytedeco.javacpp.opencv_core.CV_8UC1;

/**
 * Created by maroderi on 6/23/16.
 */
public class Debug {

    public static void print2D(BytePointer ptr, int rows, int cols, int n) {

        ByteBuffer buffer = ptr.asBuffer();

        for (int i = 0; i < n; i++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    System.out.print(String.format("%d ", buffer.get()));
                }
                System.out.println();
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void print2D(FloatPointer ptr, int rows, int cols, int n) {

        FloatBuffer buffer = ptr.asBuffer();

        for (int i = 0; i < n; i++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    System.out.print(String.format("%.2f ", buffer.get()));
                }
                System.out.println();
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void printBlob(caffe.FloatBlob blob) {

        for (int n = 0; n < blob.shape(0); n++) {
            for (int c = 0; c < blob.shape(1); c++) {
                if (blob.num_axes() > 3 && blob.shape(2) > 1) {
                    for (int x = 0; x < blob.shape(2); x++) {
                        for (int y = 0; y < blob.shape(3); y++) {
                            System.out.print(blob.data_at(n, c, x, y) + " ");
                        }
                        System.out.println();
                    }
                    System.out.println();
                } else {
                    System.out.print(blob.data_at(n, c, 0, 0) + " ");
//                    System.out.print(String.format("%.2f ", blob.data_at(n, c, 0, 0)));
                }
            }
            System.out.println();
        }
        System.out.println();
    }

    public static void saveStateAsImages(FloatPointer inputPtr, int pos, int frameSize) {
        ScreenConverter converter = new ScreenConverter();

        for (int f = 0; f < 4; f++) {
            long framePos = pos + (3 - f)*frameSize;

            opencv_core.Mat floatMat = new opencv_core.Mat(84, 84, CV_32F, (new BytePointer(inputPtr)).position(framePos * 4));
            opencv_core.Mat mat = new opencv_core.Mat(84, 84, CV_8U);
            floatMat.convertTo(mat, CV_8U, 255.0, 0);

            BufferedImage img = converter.convert(mat);

            File outputfile = new File(String.format("frame%d.png", f));
            try {
                ImageIO.write(img, "png", outputfile);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        System.out.println("SAVED FRAMES");
    }

//    public static void saveStateAsSingleChannelImage(FloatPointer inputPtr, int pos, int frameSize, int channels) {
//        ScreenConverter converter = new ScreenConverter();
//
//        byte[] imgData = new byte[frameSize*channels];
//
//        for (int f = 0; f < channels; f++) {
//            long framePos = pos + (3 - f)*frameSize;
//
//            opencv_core.Mat floatMat = new opencv_core.Mat(84, 84, CV_32F, (new BytePointer(inputPtr)).position(framePos * 4));
//            opencv_core.Mat mat = new opencv_core.Mat(84, 84, CV_8U);
//            floatMat.convertTo(mat, CV_8U, 255.0, 0);
//
//            int index = f;
//            for (int r = 0; r < 84; r++) {
//                for (int c = 0; c < 84; c++) {
//                    byte val = mat.data().get(r*84 + c);
//                    imgData[index] = val;
//                    index += channels;
//                }
//            }
//        }
//
//        opencv_core.Mat imgMat = new opencv_core.Mat(84, 84*channels, CV_8UC1);
//        imgMat.data().put(imgData);
//
//        BufferedImage img = converter.convert(imgMat);
//        File outputfile = new File("frame.jpg");
//        try {
//            ImageIO.write(img, "jpg", outputfile);
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//        System.out.println("SAVED FRAME");
//    }

    public static void saveStateAsSingleChannelImage(FloatPointer inputPtr, int pos, int frameSize, int channels) {
        ScreenConverter converter = new ScreenConverter();

        byte[] imgData = new byte[frameSize*channels];

        for (int f = 0; f < channels; f++) {
            long framePos = pos + (3 - f)*frameSize;

            opencv_core.Mat floatMat = new opencv_core.Mat(84, 84, CV_32F, (new BytePointer(inputPtr)).position(framePos * 4));
            opencv_core.Mat mat = new opencv_core.Mat(84, 84, CV_8U);
            floatMat.convertTo(mat, CV_8U, 255.0, 0);

            int index = f*frameSize;
            for (int r = 0; r < 84; r++) {
                for (int c = 0; c < 84; c++) {
                    byte val = mat.data().get(r*84 + c);
                    imgData[index] = val;
                    index ++;
                }
            }
        }

        opencv_core.Mat imgMat = new opencv_core.Mat(84*channels, 84, CV_8UC1);
        imgMat.data().put(imgData);

        BufferedImage img = converter.convert(imgMat);
        File outputfile = new File("frame.png");
        try {
            ImageIO.write(img, "png", outputfile);
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("SAVED FRAME");
    }
}
