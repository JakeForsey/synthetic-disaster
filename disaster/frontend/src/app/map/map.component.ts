import { Component, OnInit } from '@angular/core';

import OlMap from 'ol/Map';

import { MapService } from 'src/app/map/map.service';
import { BackgroundService } from 'src/app/map/background.service';
import { EditService } from 'src/app/map/edit.service';

@Component({
  selector: 'app-map',
  templateUrl: './map.component.html',
  styleUrls: ['./map.component.css']
})
export class MapComponent implements OnInit {

  map: OlMap;

  private mapService: MapService;
  private backgroundService: BackgroundService;
  private editService: EditService;

  constructor(
  	mapService: MapService,
  	backgroundService: BackgroundService,
  	editService: EditService,
  )
  {
  	this.mapService = mapService;
  	this.backgroundService = backgroundService;
  	this.editService = editService;
  }

  ngOnInit() {
      this.map = new OlMap({
      target: 'map',
      layers: [this.backgroundService.getLayer(), this.editService.getLayer()],
      view: this.mapService.view
    });
    this.editService.getInteractions().forEach( (interaction) => {
    	this.map.addInteraction(interaction);
    });
  }
}
