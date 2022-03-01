import axios from 'axios';
import './App.css';
import React,{Component} from 'react';
 
class App extends Component {
    state = {
      selectedFile: null,
      catPrediction: null,
      dogPrediction: null,
    };
    
    onFileChange = event => {    
      // Select a file from the pop up -> Update the state
      let img = event.target.files[0];
      this.setState({ selectedFile: img, image: URL.createObjectURL(img)});
    };
    
    onFileUpload = () => {
      // Click the predict button -> Create an object of formData
      const formData = new FormData();
    
      // Update the formData object
      formData.append(
        "myFile",
        this.state.selectedFile,
        this.state.selectedFile.name
      );
    
      // Details of the uploaded file
      console.log(this.state.selectedFile);
    
      // Request made to the backend api
      // Send formData object
      axios.post("http://localhost:8000/api/uploadfile", formData)
        .then(res => {
          this.setState({ catPrediction:res.data[0],dogPrediction:res.data[1]});
        })
    };
    
    // File content to be displayed after
    // file upload is complete
    fileData = () => {
    
      if (this.state.selectedFile) {
         
        return (
          <div>
            <img src={this.state.image} alt="cat or dog" />
            <h3>Prediction Results:</h3> 
            <p>File Name: {this.state.selectedFile.name}</p>
            <p>Cat prediction: {this.state.catPrediction}</p>
            <p>Dog prediction: {this.state.dogPrediction}</p>
          </div>
        );
      } 
      else {
        return (
          <div>
            <h4>Please choose an image to predict</h4>
          </div>
        );
      }
    };

    render() {
    
      return (
        <div>
            <div className='title'> Image Classifier Tool</div>
            <hr/>
            {this.fileData()}
            <hr/>
            <div>
                <input type="file" onChange={this.onFileChange}/> 
                <button onClick={this.onFileUpload}>
                  Predict
                </button>
            </div>
          
        </div>
      );
    }
  }
 
  export default App;
