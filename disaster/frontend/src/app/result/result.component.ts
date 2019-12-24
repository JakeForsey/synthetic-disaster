import { Component, OnInit } from '@angular/core';
import { ResultService } from 'src/app/result/result.service'

@Component({
  selector: 'app-result',
  templateUrl: './result.component.html',
  styleUrls: ['./result.component.css']
})
export class ResultComponent implements OnInit {

  resultService: ResultService;
  image: any;
  loadingImage: boolean;

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
      this.resultService.getImage().subscribe(data => {
        this.imageToBlob(data);
        this.loadingImage = false;
      }, error => {
        this.loadingImage = false;
        console.log(error);
      });
  }
}
