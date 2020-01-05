import { Component, OnInit } from '@angular/core';

import OlMap from 'ol/Map';
import { shiftKeyOnly } from 'ol/events/condition'

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
    let map = new OlMap({
      target: 'map',
      layers: [this.backgroundService.getLayer(), this.editService.getLayer()],
      view: this.mapService.view
    });
    map.on('click', function(event) {
      console.log("Chaning damage level.")

      if (shiftKeyOnly(event) == true) {
        let damageClasses = ["no-damage", "minor-damage", "major-damage", "destroyed"]
        map.forEachFeatureAtPixel(event.pixel, function(feature, layer) {
          let currentSubtype = feature.get("subtype");
          if (currentSubtype == undefined) {
            currentSubtype = "no-damage";
          }
          let nextSubtypeIndex = damageClasses.indexOf(currentSubtype) + 1
          if (nextSubtypeIndex > damageClasses.length) {
            nextSubtypeIndex = 0;
          }
          let nextSubtype = damageClasses[nextSubtypeIndex]
          feature.set("subtype", nextSubtype);
        });
      }
    });

    let _editService = this.editService;
    map.on('moveend', function(event) {
      _editService.getBuildingsForExtent()
    });

    this.map = map
  }
}
