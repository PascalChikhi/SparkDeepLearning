/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning;

import java.util.HashMap;
import scala.Serializable;

/**
 *
 * @author PCH
 */
public class MongoObject implements Serializable{

    private HashMap<String,String> _id;
    private String source_sector;

    public HashMap<String, String> getId() {
        return _id;
    }

    public void setId(HashMap<String, String> _id) {
        this._id = _id;
    }

    public String getSource_sector() {
        return source_sector;
    }

    public void setSource_sector(String source_sector) {
        this.source_sector = source_sector;
    }
    
    
    
}
