package dev.tcon.chapter6;

import ai.djl.*;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.*;
import ai.djl.nn.core.*;
import ai.djl.training.*;



public class ConvolutionalNetwork {


    // This method assume both array contain floats.
    public static NDArray corr2d(NDArray x, NDArray k){
        long[] hAndW = k.getShape().getShape();

        long h = hAndW[0];
        long w = hAndW[1];

        NDArray y ;
        try(NDManager manager = NDManager.newBaseManager()){
            y = manager.zeros(new Shape(x.getShape().getShape()[0] - h +1, x.getShape().getShape()[1] - w +1));


            float[][] newY = new float[(int)y.getShape().getShape()[0]][(int)y.getShape().getShape()[1]];


            for(int out = 0; out<y.getShape().getShape()[0];out++){
                for(int in = 0; in<y.getShape().getShape()[1];in++){


                    NDArray newXX = manager.create(new Shape(h,w));

                    for(int out1 = 0; out1<h; out1++){
                        for(int in1=0; in1<w;in1++){
                            newXX.set(new NDIndex(out1,in1),x.get(out+out1,in+in1));
                        }
                    }

                NDArray one =   newXX.mul(k).sum();
                newY[out][in] = (float) one.toFloatArray()[0];

                }
            }


            NDArray result = manager.create(newY);
            System.out.println(result);
            return  result;
        }









    }




    public static void smallImage(){

        try(NDManager manager = NDManager.newBaseManager()){
            float[][] floats = new float[3][3];
            float count = 0f;
            for(int i = 0;i<3;i++){
                for(int j = 0; j<3;j++){

                    floats[i][j] = count;

                    count++;


                }
            }
            count = 0f;

            float[][] kernal = new float[2][2];
            for(int i = 0;i<2;i++) {
                for (int j = 0; j < 2; j++) {
                    kernal[i][j] = count;

                    count++;

                }
            }

            NDArray X = manager.create(floats);


            NDArray K = manager.create(kernal);

            corr2d(X,K);


        }

    }

    public static void largeImage(){

        Application application = Application.CV.IMAGE_CLASSIFICATION;

        long inputSize = 6*8;

        long outputSize = 4;


        try(NDManager manager = NDManager.newBaseManager()){

            float[][] intArray = new float[6][8];

            for(int i=0 ; i< 6;i++ ){
                for(int j=0; j<8; j++){

                    if( j>1 && j<6){
                        intArray[i][j] = 0;
                    }else {
                        intArray[i][j] = 1;
                    }

                }
            }

            float[][] kernalArray = new float[1][2];

            kernalArray[0][0] = 1;
            kernalArray[0][1] = -1;

            NDArray kernel = manager.create(kernalArray);


            NDArray nd = manager.create(intArray);


           corr2d(nd,kernel);
           corr2d(nd.transpose(),kernel);






//            System.out.println(nd);
//            System.out.println(kernel);

        }
    }


    public static void main(String[] args) {

        largeImage();


        smallImage();





        }












}
