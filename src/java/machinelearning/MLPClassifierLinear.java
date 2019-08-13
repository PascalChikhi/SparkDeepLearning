package machinelearning;

import com.mongodb.spark.MongoSpark;
import com.mongodb.spark.config.ReadConfig;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.bson.Document;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * "Linear" Data Classification Example
 *
 * Based on the data from Jason Baldridge:
 * https://github.com/jasonbaldridge/try-tf/tree/master/simdata
 *
 * @author Josh Patterson
 * @author Alex Black (added plots)
 *
 */
public class MLPClassifierLinear {

private static final Logger log = LoggerFactory.getLogger(MLPClassifierLinear.class);    

    public static void main(String[] args) throws Exception {
        int seed = 123;
        double learningRate = 0.01;
        int batchSize = 891;
        int nEpochs = 30;
        int batchSizePerWorker = 5;
    
        int numInputs = 5;
        int numOutputs = 2;
        int numHiddenNodes = 20;
        
        SparkConf sparkConf = new SparkConf()
                .setAppName("JavaSparkMongoExample")
                .setMaster("local")
                .set("spark.mongodb.input.uri", "mongodb://127.0.0.1/test.train_master")
                .set("spark.mongodb.output.uri", "mongodb://127.0.0.1/test.train_master");
        
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);        
       
//        double[][] mylabel = {{0,1},{1,0},{1,0},{1,0},{1,0}};
//        double[][] mytest = {{1,2,3,4,5},{1,1,1,1,1},{0,0,0,0,0},{10,20,30,40,50},{1,2,1,2,1}};
        
//        double[] mylabel2 = {1,0};
//        double[] mytest2 = {1.0,1.0,1.0,1.0,1.0};        
        
//        INDArray myNDArray = Nd4j.create(mytest);        
//        INDArray myNDArray2 = Nd4j.create(mylabel); 
        
//        DataSet myds1 = new DataSet(myNDArray,myNDArray2);
        
//        INDArray myNDArray3 = Nd4j.create(mytest2);        
//        INDArray myNDArray4 = Nd4j.create(mylabel2); 
        
//        DataSet myds2 = new DataSet(myNDArray3,myNDArray4);        
        
//        INDArray alpha = myds1.getFeatures();
//        INDArray beta = myds1.getLabels();
        
//        DataSetIterator trainIter = new ListDataSetIterator(myds1.asList(),batchSize);
     
//        DataSetIterator testIter = new ListDataSetIterator(myds2.asList(),batchSize);

//        ---------------------------------------------------------------------------------------       

//        Load the training data:
//        RecordReader rr = new CSVRecordReader();
//        rr.initialize(new FileSplit(new File("src/resources/train_master.csv")));
//
//        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr,batchSize,0,2);
//        
////        //Load the test/evaluation data:
//        RecordReader rrTest = new CSVRecordReader();
//        rrTest.initialize(new FileSplit(new File("src/resources/test_master.csv")));
//        
//        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest,batchSize,0,2);
//        
//        List<DataSet> trainDataList = new ArrayList<>();
//
//        while (trainIter.hasNext()) {
//            trainDataList.add(trainIter.next());
//        }
//
//        List<DataSet> testDataList = new ArrayList<>();       
//        
//        while (testIter.hasNext()) {
//            testDataList.add(testIter.next());
//        }
        
//    ---------------------------------------------------------------------------        
        
        SparkConnect a = new SparkConnect();
        JavaRDD<DataSet> rdd_train_master = a.get_mongo_rdd2(jsc,"train_master");
        JavaRDD<DataSet> rdd_test_master = a.get_mongo_rdd2(jsc,"test_master");
        
//        JavaRDD<DataSet> trainData = jsc.parallelize(trainDataList);
//        JavaRDD<DataSet> testData = jsc.parallelize(testDataList);
//        JavaRDD<String> test_csv = sc.textFile("src/resources/test_master.csv");

//       for(DataSet d : trainData.collect()){
//            System.out.println(d.toString());
//        }                 
//       
//        for(DataSet d : rdd_train_master.collect()){
//            System.out.println(d.toString());
//        }

        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(1)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax").weightInit(WeightInit.XAVIER)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();


//        MultiLayerNetwork model = new MultiLayerNetwork(conf);
//        model.init();
//        model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates
//
//        for ( int n = 0; n < nEpochs; n++) {
//            model.fit( trainIter );
//
//        }

        //Configuration for Spark training: see http://deeplearning4j.org/spark for explanation of these configuration options
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)    //Each DataSet object: contains (by default) 32 examples
            .averagingFrequency(5)
            .workerPrefetchNumBatches(2)            //Async prefetching: 2 examples per worker
            .batchSizePerWorker(batchSizePerWorker)
            .build();

        //Create the Spark network
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(jsc, conf, tm);
        MultiLayerNetwork trainedNetwork = null;
        
        //Execute training:
        for (int i = 0; i < nEpochs; i++) {
//          trainedNetwork =  sparkNet.fit(trainData);
        trainedNetwork =  sparkNet.fit(rdd_train_master);
        log.info("Completed Epoch {}", i);
        }

        //Perform evaluation (distributed)
//          Evaluation evaluation = sparkNet.evaluate(testData);        
        Evaluation evaluation = sparkNet.evaluate(rdd_test_master);
        log.info("***** Evaluation *****");
        log.info(evaluation.stats());

        //Delete the temp training files, now that we are done with them
        tm.deleteTempFiles(jsc);

        log.info("***** Example Complete *****");        
              
        // 1st class,Male,52 yrs,71.25,Southampton
        double[] mytest2 = {1,1,52,71.25,0};

//        LabeledPoint pos = new LabeledPoint(1.0, Vectors.dense(mytest2));   
//        
//        List<LabeledPoint> mylist = new ArrayList<>();
//        mylist.add(pos);
//        
//        JavaRDD<LabeledPoint> testData2 = sc.parallelize(mylist);
//        
//        JavaRDD<DataSet> testData3 = MLLibUtil.fromLabeledPoint(sc, testData2, 2);
//        
//        Evaluation eval1 = sparkNet.evaluate(testData3);
          
       Vector v1 = Vectors.dense(mytest2);
        
       Vector res1 =  sparkNet.predict(v1);
        System.out.println(res1); 
//        INDArray gamma = model.output(myNDArray3,false);
//        System.out.println(gamma.toString());
        
        System.out.println("Evaluate model....");

     
//        JavaRDD<LabeledPoint> testData4 = test_csv.map(new Function<String, LabeledPoint>() {
//            @Override
//            public LabeledPoint call(String line) throws Exception {
//            String[] array = line.split(",");
//            for(int i=0;i<array.length;i++){
//               if(array[i].isEmpty()){
//                   array[i]="0.0";
//               }
//            }
//            String array_label = Arrays.copyOfRange(array, 0, 1)[0];
//            String[] array_vector = Arrays.copyOfRange(array, 1, array.length);
//
//            double[] doubleValues = Arrays.stream(array_vector)
//                                    .mapToDouble(Double::parseDouble)
//                                    .toArray();
//            double label = Double.valueOf(array_label);
//            Vector vector = Vectors.dense(doubleValues);
//          
//            return new LabeledPoint(label,vector );                
//            }
//        });
//        
//        JavaRDD<DataSet> testData5 = MLLibUtil.fromLabeledPoint(sc, testData4, 2);
//                
//        Evaluation eval2 = sparkNet.evaluate(testData5);
                
//        Evaluation eval = new Evaluation(numOutputs);        
//        while(testIter.hasNext()){
//            DataSet t = testIter.next();
//            INDArray features = t.getFeatureMatrix();
//            INDArray lables = t.getLabels();
//            INDArray predicted = model.output(features,false);
//
//            eval.eval(lables, predicted);
//
//        }

        //Print the evaluation statistics
//        System.out.println(eval2.stats());
      

        System.out.println("****************Example finished********************");
        

        //Save the model
        File locationToSave = new File("C:/Users/PCH/Desktop/MyMultiLayerNetwork.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(trainedNetwork, locationToSave, saveUpdater);

        //Load the model
        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(locationToSave);        
        int yy=0;
        
    }
}
