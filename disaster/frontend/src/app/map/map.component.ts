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

  private damageClasses: string[];

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
    this.editService.getInteractions().forEach( (interaction) => {
      map.addInteraction(interaction);
    });

    map.on('click', function(event) {
      if (shiftKeyOnly(event) == true) {
        let damageClasses = ["no-damage", "minor-damage", "major-damage", "destroyed"]
        map.forEachFeatureAtPixel(event.pixel, function(feature, layer) {
          let currentSubtype = feature.get("subtype");
          if (currentSubtype == undefined) {
            currentSubtype = "no-damage";
          }
          let nextSubtype = damageClasses[damageClasses.indexOf(currentSubtype) + 1]

          console.log(nextSubtype)
          feature.set("subtype", nextSubtype);
        });
      }
    });

    this.map = map
  }
  incrementDamage(feature) {
  }

}
