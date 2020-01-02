import { Injectable } from '@angular/core';

import WKT from 'ol/format/WKT';
import OlVector from 'ol/source/Vector';
import OlVectorLayer from 'ol/layer/Vector';
import OlDraw from 'ol/interaction/Draw';
import OlModify from 'ol/interaction/Modify';
import OlSnap from 'ol/interaction/Snap';
import Select from 'ol/interaction/Select';
import { click, shiftKeyOnly } from 'ol/events/condition';

import { MapService } from 'src/app/map/map.service';

interface Properties {
  feature_type: string;
  subtype: string;
  uuid: string;
}
interface XView2Feature {
  properties: Properties;
  wkt: string;
}
interface Features {
  xy: XView2Feature[];
}
interface Layout {
  features: Features;
  scene: string;
}

@Injectable({
  providedIn: 'root'
})
export class EditService {

  private mapService: MapService;

  private source: OlVector;
  private layer: OlVectorLayer;

  private draw: OlDraw;
  private modify: OlModify;
  private snap: OlSnap;

  private interactions: Array<any>;
  private features: any[] = [];

  private wktFormat: any;

  constructor(
    	mapService: MapService,
  ) {
    this.mapService = mapService;

    this.source = new OlVector({});

    this.layer = new OlVectorLayer({
      source: this.source
    });

    this.draw = new OlDraw({
      source: this.source,
      type: "Polygon",
      features: this.features,
      condition: function(e) {
        if (e.pointerEvent.buttons === 1) {
          return true;
        } else {
          return false;
        }
      }
    });

    this.modify = new OlModify({source: this.source});
    this.snap = new OlSnap({source: this.source});

    this.interactions = [this.draw, this.modify, this.snap];

    this.wktFormat = new WKT()
  }

  getLayer() {
    return this.layer;
  }
  getInteractions() {
    return this.interactions;
  }
  getLayout() {
    let myfeatures: XView2Feature[] = new Array();
    let extent = this.mapService.view.calculateExtent()

    let width = Math.abs(extent[0] - extent[2])
    let height = Math.abs(extent[1] - extent[3])
    let widthResolution = width / 512
    let heightResolution = height / 512

    this.draw.features_.forEach( (feature) => {
      let geometry = feature.getGeometry().clone();
      let coordinates: any[] = new Array();

      feature.getGeometry().getCoordinates().forEach( (ringCoordinates) => {
        ringCoordinates.forEach( (coordinate) => {
          let newCoordinate = [Math.abs(coordinate[0] - extent[0]) / widthResolution, Math.abs(coordinate[1] - extent[1]) / heightResolution];
          coordinates.push(newCoordinate)
        });
      });
      // hard coded single ring polygon
      geometry.setCoordinates([coordinates]);

      myfeatures.push(
        {
          properties: {
            feature_type: "building",
            subtype: feature.get("subtype"),
            uuid: "asdagagas",
          },
          wkt: this.wktFormat.writeGeometry(
             geometry
          )
        }
      );
    });

    return {features: {xy: myfeatures}, scene: "socal-fire"};
  }
}
