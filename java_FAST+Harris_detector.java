import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.image.BufferedImage;
import java.awt.image.MemoryImageSource;
import java.awt.image.PixelGrabber;
import java.io.File;
import java.io.IOException;
import java.net.URL;

/**
 * Created with IntelliJ IDEA.
 * User: trane91
 * Date: 4/18/13
 * Time: 12:02 AM
 * To change this template use File | Settings | File Templates.
 */
public class Main {

    private static final String TAG = Main.class.getSimpleName();

    public static void main(String[] args){

        if( args.length < 2){
            Log.d(TAG, "Image address is null. Please enter image address.");
            return;
        }

        String imageAddress = args[0];
        BufferedImage image;

        try {
            image = ImageIO.read(new File( imageAddress).toURI().toURL());
        } catch (IOException e) {
            //e.printStackTrace();
            Log.d(TAG, "Read image error.");
            return;
        }
        BufferedImage destonationImage = image;

        int[][] lightnessArray = new int[image.getHeight()][image.getWidth()];
        Log.d(TAG, "Initialize intensity.");

        for(int y=0; y<image.getHeight(); y++)
            for(int x=0; x<image.getWidth();x++){
                int []rgb = getPixelData( image, x, y);
                lightnessArray[y][x] = getIntensity(rgb);
            }
        Log.d(TAG, "Finish initialize intensity.");

        int threshold = Integer.parseInt( args[1]);
        // 8 связный пиксель
        /*for(int y=1; y < image.getHeight()-1; y++)
            for(int x=1; x < image.getWidth()-1;x++){

                int leftUpCornerAverageLightness = (lightnessArray[y][x-1] +
                        lightnessArray[y-1][x-1] + lightnessArray[y-1][x]) / 3;

                int rightUpCornerAverageLightness = (lightnessArray[y][x+1] +
                        lightnessArray[y-1][x+1] + lightnessArray[y-1][x]) / 3;

                int leftDownCornerAverageLightness = (lightnessArray[y][x-1] +
                        lightnessArray[y+1][x-1] + lightnessArray[y+1][x]) / 3;

                int rightDownCornerAverageLightness = (lightnessArray[y][x+1] +
                        lightnessArray[y+1][x+1] + lightnessArray[y+1][x]) / 3;

                int leftDifference = Math.abs( leftUpCornerAverageLightness - leftDownCornerAverageLightness);
                int rightDifference = Math.abs( rightUpCornerAverageLightness - rightDownCornerAverageLightness);
                int upDifference = Math.abs( leftUpCornerAverageLightness - rightUpCornerAverageLightness);
                int downDifference = Math.abs( leftDownCornerAverageLightness - rightDownCornerAverageLightness);

                if ( leftDifference >= threshold || rightDifference >= threshold){
                    if( upDifference >= threshold || downDifference >= threshold){
                        Log.d(TAG, "Diff. x: " + x + " y:" + y);
                        drawPixel( destonationImage.getGraphics(), x, y, 3, Color.RED);
                    }
                }
            }*/

        int detected[][] = new int[image.getHeight()][image.getWidth()];

        int maxLightness = 0;
        int minLightness = 255;
        int detectedCounter =0;

        //16 связный пиксель
        for(int y=1; y < image.getHeight()-1; y++)
            for(int x=1; x < image.getWidth()-1;x++){

                if( maxLightness < lightnessArray[y][x])
                    maxLightness = lightnessArray[y][x];
                if( minLightness > lightnessArray[y][x])
                    minLightness = lightnessArray[y][x];

                int thresholdCounter=0;
                int centerLightness = lightnessArray[y][x];
                /*//Up line
                if( lightnessArray[y-1][x-1] - centerLightness > threshold)
                    thresholdCounter++;
                if( lightnessArray[y-1][x] - centerLightness > threshold)
                    thresholdCounter++;
                if( lightnessArray[y-1][x+1] - centerLightness > threshold)
                    thresholdCounter++;

                //Right line
                if( lightnessArray[y-1][x+1] - centerLightness > threshold)
                    thresholdCounter++;
                if( lightnessArray[y][x+1] - centerLightness > threshold)
                    thresholdCounter++;
                if( lightnessArray[y+1][x+1] - centerLightness > threshold)
                    thresholdCounter++;

                //Down line
                if( lightnessArray[y+1][x+1] - centerLightness > threshold)
                    thresholdCounter++;
                if( lightnessArray[y+1][x] - centerLightness > threshold)
                    thresholdCounter++;
                if( lightnessArray[y+1][x-1] - centerLightness > threshold)
                    thresholdCounter++;

                //Left line
                if( lightnessArray[y-1][x-1] - centerLightness > threshold)
                    thresholdCounter++;
                if( lightnessArray[y][x-1] - centerLightness > threshold)
                    thresholdCounter++;
                if( lightnessArray[y+1][x-1] - centerLightness > threshold)
                    thresholdCounter++;

                if( thresholdCounter > 2){

                }*/

                int leftLineAverageLightness = (lightnessArray[y-1][x-1] + lightnessArray[y][x-1] + lightnessArray[y+1][x-1]) / 3;
                int upLineAverageLightness = (lightnessArray[y-1][x-1] + lightnessArray[y-1][x] + lightnessArray[y-1][x+1]) / 3;
                int rightLineAverageLightness = (lightnessArray[y-1][x+1] + lightnessArray[y][x+1] + lightnessArray[y+1][x+1]) / 3;
                int downLineAverageLightness = (lightnessArray[y+1][x+1] + lightnessArray[y+1][x] + lightnessArray[y+1][x-1]) / 3;

                int leftUpCornerAverageLightness = ( leftLineAverageLightness + upLineAverageLightness ) / 2;
                int rightUpCornerAverageLightness = ( upLineAverageLightness + rightLineAverageLightness ) / 2;
                int leftDownCornerAverageLightness = ( leftLineAverageLightness + downLineAverageLightness ) / 2;
                int rightDownCornerAverageLightness = ( rightLineAverageLightness + downLineAverageLightness ) / 2;

                int leftDifference = Math.abs( leftUpCornerAverageLightness - leftDownCornerAverageLightness);
                int rightDifference = Math.abs( rightUpCornerAverageLightness - rightDownCornerAverageLightness);
                int upDifference = Math.abs( leftUpCornerAverageLightness - rightUpCornerAverageLightness);
                int downDifference = Math.abs( leftDownCornerAverageLightness - rightDownCornerAverageLightness);

                detected[y][x] = 0;
                if ( leftDifference >= threshold || rightDifference >= threshold)
                    if( upDifference >= threshold || downDifference >= threshold){
                        detected[y][x] = 1;
                        detectedCounter++;
                        drawPixel( destonationImage.getGraphics(), x, y, 3, Color.RED);
                    }

                if( detected[y][x] != 1 &&  (upDifference >= threshold || downDifference >= threshold) )
                    if ( leftDifference >= threshold || rightDifference >= threshold){
                        detected[y][x] = 1;
                        detectedCounter++;
                        drawPixel( destonationImage.getGraphics(), x, y, 3, Color.RED);
                    }

            }

        /*int drawPixelCounter = 0;
        for(int y=1; y < image.getHeight()-1; y++)
            for(int x=1; x < image.getWidth()-1;x++){
                if( detected[y][x] == 1){
                    *//*boolean flag = true;
                    Log.d(TAG, "Detected x = " + x + " y = " + y);
                    for(int y1 = y-1; y1 <= y; y1++)
                        for(int x1 = x-1; x1 <= x+1; x1++)
                            if( detected[y1][x1] == 1){
                                flag = false;
                                break;
                            }

                    if(flag){

                        drawPixelCounter++;
                        drawPixel( destonationImage.getGraphics(), x, y, 3, Color.RED);
                    }*//*

                    if( detected[y-1][x-1] == 1 ||
                        detected[y-1][x] == 1 ||
                        detected[y-1][x+1] == 1 ||
                        detected[y][x-1] == 1 ||
                        detected[y][x+1] == 1){
                        drawPixelCounter++;
                        drawPixel( destonationImage.getGraphics(), x, y, 3, Color.RED);
                    }
                }
            }*/

        Log.d(TAG, "Max lighntess: " + maxLightness + " Min lightness: " + minLightness + " Detected counter: " + detectedCounter);

        //Log.d(TAG, "Draw pixel counter: " + drawPixelCounter);

        JFrame frame = new JFrame();
        frame.setSize(image.getWidth(),image.getHeight());
        frame.setDefaultCloseOperation( JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(new JLabel(new ImageIcon( destonationImage)));
        frame.show();


    }

    public static void drawPixel(Graphics g, int x, int y, int size, Paint color){
        Graphics2D ga = (Graphics2D)g;
        Shape circle = new Ellipse2D.Float(x, y, size, size);
        ga.setPaint(color);
        ga.draw(circle);
        ga.setPaint(color);
        ga.fill(circle);
    }

    /*
    *  int[][] array = {
    { 79, 85, 73},
    { 126, 132, 120},
    { 127, 133, 121},
    { 126, 132, 122},
    { 144, 150, 140},
    { 140, 146, 136},
    { 143, 151, 140},
    { 146, 154, 143},
    { 127, 135, 124},
    { 128, 136, 125},
    { 128, 136, 125},
    { 78, 86, 75}};*/

    /*
    * int[][] array2 = {
                { 91, 100, 83},
                { 93, 102, 85},
                { 51, 103, 88},

                { 94, 102, 87},
                { 78, 84, 72},
                { 79, 85, 73},

                { 78, 86, 75},
                { 77, 85, 74},
                { 49, 57, 46},

                { 50, 58, 45},
                { 50, 58, 43},
                { 90, 97, 81}
        };*/
    public static void res(int [][] array){
        int length = array.length;
        StringBuilder sb =new StringBuilder();
        sb.append("Lightness array: ");
        int max =0, min = 255, sum = 0;

        for(int i=0; i < length; i++){
            sb.append( getLightness( array[i]) + ",");
            if( min > getLightness( array[i]))
                min = getLightness( array[i]);
            if( max < getLightness( array[i]))
                max = getLightness( array[i]);

            sum += getLightness( array[i]);
        }
        sum = sum / array.length;
        Log.d(TAG, sb.toString());
        Log.d(TAG, "Max: " + max + " Min: " + min + " Threshold: " + (max - min) + " Sum: " + sum);

        sb.setLength(0);
        sb.append("Intensity array: ");
        max = 0;
        min = 255;
        sum = 0;
        for(int i=0; i < length; i++){
            sb.append( getIntensity( array[i]) + ",");
            if( min > getIntensity( array[i]))
                min = getIntensity( array[i]);
            if( max < getIntensity( array[i]))
                max = getIntensity( array[i]);

            sum += getIntensity( array[i]);
        }
        sum = sum / array.length;
        Log.d(TAG, sb.toString());
        Log.d(TAG, "Max: " + max + " Min: " + min + " Threshold: " + (max - min) + " Sum: " + sum);

        sb.setLength(0);
        sb.append("GS array: ");
        max = 0;
        min = 255;
        sum = 0;
        for(int i=0; i < length; i++){
            sb.append( getGS( array[i]) + ",");
            if( min > getGS( array[i]))
                min = getGS( array[i]);
            if( max < getGS( array[i]))
                max = getGS( array[i]);
            sum += getGS( array[i]);
        }
        sum = sum / array.length;
        Log.d(TAG, sb.toString());
        Log.d(TAG, "Max: " + max + " Min: " + min + " Threshold: " + (max - min) + " Sum: " + sum);
    }

    public static int getIntensity(int []rgb){
        return (int)(0.3*rgb[0] + 0.59*rgb[1] + 0.11*rgb[2]);
    }

    public static int getGS(int []rgb){
        return (int)(0.21*rgb[0] + 0.72*rgb[1] + 0.07*rgb[2]);
    }

    public static int getLightness(int []rgb){
        int max = Math.max( rgb[0], Math.max( rgb[1], rgb[2]));
        int min = Math.min(rgb[0], Math.min(rgb[1], rgb[2]));
        return (int)( (max + min) / 2);
    }

    private static int[] getPixelData(BufferedImage img, int x, int y) {
        int argb = img.getRGB(x, y);

        int rgb[] = new int[] {
                (argb >> 16) & 0xff, //red
                (argb >>  8) & 0xff, //green
                (argb      ) & 0xff  //blue
        };
        return rgb;
    }

    /*
    * if(args.length < 1){
            Log.d(TAG, "Enter image.");
            return;
        }

        String imageAddress = args[0];
        try {
            BufferedImage image = ImageIO.read(new File( imageAddress).toURI().toURL());
            image = getGrayScale( image);

            /*int []rgb = getPixelData( image, 128, 62);
            Log.d(TAG, "Pixel 128,62: " + rgb[0] + " " + rgb[1] + " " + rgb[2]);
            rgb = getPixelData( image, 129, 62);
            Log.d(TAG, "Pixel 129,62: " + rgb[0] + " " + rgb[1] + " " + rgb[2]);
            Log.d(TAG, " Height: " + image.getHeight() + " Width: " + image.getWidth());

        int []orig = new int[ image.getWidth() * image.getHeight()];
        PixelGrabber grabber = new PixelGrabber( image, 0, 0,
                image.getWidth(), image.getHeight(), orig, 0, image.getWidth());

        try {
            grabber.grabPixels();
        }
        catch(InterruptedException e2) {
            System.out.println("error: " + e2);
        }

        harris h = new harris();
        h.init( orig, image.getWidth(), image.getHeight(), 0.001);
        orig=h.process();

        //final Image output = createImage(new MemoryImageSource(width, height, orig, 0, width));
        final Image output =  Toolkit.getDefaultToolkit().
                createImage(new MemoryImageSource(image.getWidth(), image.getHeight(), orig, 0, image.getWidth()));
        JFrame frame = new JFrame();
        frame.setSize(500,500);
        frame.setDefaultCloseOperation( JFrame.EXIT_ON_CLOSE);
        frame.getContentPane().add(new JLabel(new ImageIcon(output)));
        frame.show();


        } catch (IOException e) {
                e.printStackTrace();
        Log.d(TAG, "Image read error.");
        }
    * */
}
