/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning;


import com.google.gson.Gson;
import com.mongodb.spark.MongoSpark;
import com.mongodb.spark.config.ReadConfig;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.bson.Document;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import scala.Serializable;

/**
 *
 * @author PCH
 */
public class SparkConnect implements Serializable{
 
public JavaRDD<DataSet>  get_mongo_rdd2(JavaSparkContext jsc,String mycollection){
        
//    SparkSession spark = SparkSession
//      .builder()
//      .master("local")      
//      .appName("MongoSparkConnectorIntro")
//      .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/test.train_master")
//      .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/test.train_master")            
//      .getOrCreate();    
  
//    JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());   
//        SparkConf sparkConf = new SparkConf()
//                .setAppName("JavaSparkMongoExample")
//                .setMaster("local");
//        
//    JavaSparkContext jsc = new JavaSparkContext(sparkConf);  
    
    Map<String, String> readOverrides = new HashMap<>();
    readOverrides.put("collection", mycollection);
    
    ReadConfig readconfig = ReadConfig.create(jsc).withOptions(readOverrides);
       
    JavaRDD<Document> rdd_doc = MongoSpark.load(jsc, readconfig);

    INDArray feature = null ;
    INDArray label = null ;
    int k=0;
    
    List<double[]> feature_list = new ArrayList<>();
    List<double[]> label_list = new ArrayList<>();
    
    for(Document d: rdd_doc.collect()){
    
    Gson gson = new Gson();
    String gsonString = d.toJson();    
    MongoObject mongo_object = gson.fromJson(gsonString, MongoObject.class);    

    
    String[] my_array  = mongo_object.getSource_sector().split(",");
                double[] vect_feature = new double[my_array.length-1];
                double[] vect_label = new double[2];
                String temp="";
                double nb=0.0,nb_compl=0.0;
                int size = my_array.length;
                
                for(int i=0;i<size;i++){
                
                temp = my_array[i];    
                    
                if(temp.isEmpty() || temp.equals("")){
                          temp="0.0";
                      }
           
                nb = Double.valueOf(temp);
                
                if(i==0){
                      if(nb==0.0){
                        nb_compl=1.0;  
                      } else {
                        nb_compl=0.0;    
                      }
                      vect_label[i] = nb;
                      vect_label[i+1] = nb_compl;                      
                  } else {
                      vect_feature[i-1] = Double.valueOf(temp);
                  } 
                }
            feature_list.add(vect_feature);
            label_list.add(vect_label);
            k++;                    
            }
    
    double[][] feature_array = new double[feature_list.size()][];
    feature_array = feature_list.toArray(feature_array);

    double[][] label_array = new double[label_list.size()][];
    label_array = label_list.toArray(label_array);
    
    feature = Nd4j.create(feature_array);
    label = Nd4j.create(label_array);
    
    DataSet dataset = new DataSet(feature,label);
    List<DataSet> mylist = new ArrayList<>();
    mylist.add(dataset);
    
    JavaRDD<DataSet> rdd = jsc.parallelize(mylist);
    
    return rdd;
}

@Deprecated            
public JavaRDD<DataSet> get_mongo_rdd(String mycollection){
        
//    SparkSession spark = SparkSession
//      .builder()
//      .master("local")      
//      .appName("MongoSparkConnectorIntro")
//      .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/test.train_master")
//      .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/test.train_master")            
//      .getOrCreate();    
          SparkConf sparkConf = new SparkConf()
                .setAppName("JavaSparkMongoExample")
                .setMaster("local");
          
    JavaSparkContext jsc = new JavaSparkContext(sparkConf);   
    
    Map<String, String> readOverrides = new HashMap<>();
    readOverrides.put("collection", mycollection);
    
    ReadConfig readconfig = ReadConfig.create(jsc).withOptions(readOverrides);    
   
    JavaRDD<DataSet> rdd = MongoSpark.load(jsc, readconfig).map(new Function<Document, DataSet>() {
            @Override
            public DataSet call(Document t1) throws Exception {
             Gson gson = new Gson();
            String gsonString = t1.toJson();    
            MongoObject mongo_object = gson.fromJson(gsonString, MongoObject.class);    
                
                
                String[] my_array  = mongo_object.getSource_sector().split(",");
                double[] vect_feature = new double[my_array.length-1];
                double[] vect_label = new double[1];
                String res="",temp="";
                int size = my_array.length;
                
                for(int i=0;i<size;i++){
                  if(i==0){
                      res = my_array[i];
                      vect_label[i] = Double.valueOf(res);
                  } else {
                      temp = my_array[i];
                      if(temp.isEmpty() || temp.equals("")){
                          temp="0.0";
                      }
                      vect_feature[i-1] = Double.valueOf(temp);
                  } 
                }
                
                INDArray feature = Nd4j.create(vect_feature);
                INDArray label = Nd4j.create(vect_label);
                
                DataSet dataset = new DataSet(feature,label);
                
                return dataset;
                
            }
        });    
    
    return rdd;
}
    
    
    
}
