import { Component, OnInit } from '@angular/core';
import { ResultService } from 'src/app/result/result.service'
import { MapService } from 'src/app/map/map.service'

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
  mapService: MapService;

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
    {value: 'woolsey-fire', viewValue: 'Woolsey fire'},
    {value: 'pinery-bushfire', viewValue: 'Pinery Bushfire'},
    {value: 'portugal-wildfire', viewValue: 'Portugal wildfire'},
    {value: 'nepal-flooding', viewValue: 'Nepal flooding'},
    {value: 'sunda-tsunami', viewValue: 'Sunda tsunami'},
    {value: 'lower-puna-volcano', viewValue: 'Lower Puna volcano'},
    {value: 'moore-tornado', viewValue: 'Moore tornado'},
    {value: 'joplin-tornado', viewValue: 'Joplin tornado'},
    {value: 'tuscaloosa-tornado', viewValue: 'Tuscaloosa tornado'},
  ];
  selectedScene: string = "hurricane-michael";

  constructor(resultService: ResultService, mapService: MapService) {
    this.resultService = resultService;
    this.mapService = mapService;
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

      let bounds = this.mapService.view.calculateExtent()
      let minLon = bounds[0]
      let minLat = bounds[1]
      let maxLon = bounds[2]
      let maxLat = bounds[3]
  
      this.resultService.getImage(this.selectedScene, minLon, minLat, maxLon, maxLat).subscribe(data => {
        this.imageToBlob(data);
        this.loadingImage = false;
      }, error => {
        this.loadingImage = false;
        console.log(error);
      });
  }
}
