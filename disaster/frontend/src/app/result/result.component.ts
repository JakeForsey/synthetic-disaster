import { Component, OnInit } from '@angular/core';
import { ResultService } from 'src/app/result/result.service'

interface Scene {
  value: string
  viewValue: string
}
@Component({
  selector: 'app-result',
  templateUrl: './result.component.html',
  styleUrls: ['./result.component.css']
})
export class ResultComponent implements OnInit {

  resultService: ResultService;
  image: any;
  loadingImage: boolean;
  scenes: Scene[] = [
    {value: 'hurricane-michael', viewValue: 'Hurricane Michael'},
    {value: 'hurricane-matthew', viewValue: 'Hurricane Matthew'},
    {value: 'hurricane-florence', viewValue: 'Hurricane Florence'},
    {value: 'hurricane-harvey', viewValue: 'Hurricane Harvey'},
    {value: 'socal-fire', viewValue: 'Socal fire'},
    {value: 'santa-rosa-wildfire', viewValue: 'Santa-Rosa wildfire'},
    {value: 'guatemala-volcano', viewValue: 'Guatemala volcano'},
    {value: 'palu-tsunami', viewValue: 'Palu tsunami'},
    {value: 'midwest-flooding', viewValue: 'Midwest flooding'},
    {value: 'mexico-earthquake', viewValue: 'Mexico earthquake'},
  ];
  selectedScene: string;

  constructor(resultService: ResultService) {
    this.resultService = resultService;
  }

  ngOnInit() {
  }

  imageToBlob(image: Blob) {
     let reader = new FileReader();
     reader.addEventListener("load", () => {
        this.image = reader.result;
     }, false);

     if (image) {
        reader.readAsDataURL(image);
     }
  }

  onClick() {
      this.loadingImage = true;
      this.resultService.scene = this.selectedScene;

      this.resultService.getImage().subscribe(data => {
        this.imageToBlob(data);
        this.loadingImage = false;
      }, error => {
        this.loadingImage = false;
        console.log(error);
      });
  }
}
