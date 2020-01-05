import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { HttpParams } from "@angular/common/http";

import WKT from 'ol/format/WKT';
import OlVector from 'ol/source/Vector';
import OlVectorLayer from 'ol/layer/Vector';
import { transformExtent } from 'ol/proj';
import OlDraw from 'ol/interaction/Draw';
import OlModify from 'ol/interaction/Modify';
import OlSnap from 'ol/interaction/Snap';
import GeoJSON from 'ol/format/GeoJSON';
import { Style, Fill } from 'ol/style';
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
      private http: HttpClient,
    	mapService: MapService,
  ) {
    this.mapService = mapService;

    this.source = new OlVector({});
    
    this.layer = new OlVectorLayer({
      source: this.source,
      style: function(feature, resolution) {
        let subtype = feature.get("subtype");
        let colour = [0, 0, 0, 0.5];
        switch (subtype) {
          case undefined: {
            colour = [0, 0, 0, 0.5]
          }
          case "no-damage": {
            colour = [0, 0, 0, 0.5]
            break;
          }
          case "minor-damage": {
            colour = [60, 0, 0, 0.5]
            break;
          }
          case "major-damage": {
            colour = [120, 0, 0, 0.5]
            break;
          }
          case "destroyed": {
            colour = [180, 0, 0, 0.5]
            break;
          }
        }
        return [
          new Style({
            fill: new Fill({
              color: colour
            })
          })
        ];
      }
    });

    this.wktFormat = new WKT()
    console.log("Edit service intitialised.")
  }

  public getLayer() {
    return this.layer;
  }

  public getInteractions() {
    return this.interactions;
  }

  public getBuildingsForExtent() {
    console.log("Clearing buildings.")
    this.source.clear();

    console.log("Fetching OSM buildings for map extent.")
    let bounds = this.mapService.view.calculateExtent()
    bounds = transformExtent(bounds, 'EPSG:3857','EPSG:4326');
    let minLon = bounds[0]
    let minLat = bounds[1]
    let maxLon = bounds[2]
    let maxLat = bounds[3]

    let osmBuildings = this.http.get(
      "http://localhost:6001/osm",
      { params: new HttpParams().set('minLon', minLon).set('minLat', minLat).set('maxLon', maxLon).set('maxLat', maxLat) },
    );

    console.log("Adding OSM buildings to the map.")
    osmBuildings.subscribe(geojsonObject => {
      let features = (new GeoJSON()).readFeatures(geojsonObject, {
        dataProjection : 'EPSG:4326',
        featureProjection: 'EPSG:3857'
      });
      this.source.addFeatures(features);
    }, error => {
      console.log(error);
    });
  }

  public getLayout() {
    console.log("Generating scene layout.")

    let myfeatures: XView2Feature[] = new Array();
    let extent = this.mapService.view.calculateExtent()

    let width = Math.abs(extent[0] - extent[2])
    let height = Math.abs(extent[1] - extent[3])
    let widthResolution = width / 512
    let heightResolution = height / 512

    this.source.getFeatures().forEach( (feature) => {
      let geometry = feature.getGeometry().clone();
      let coordinates: any[] = new Array();

      feature.getGeometry().getCoordinates().forEach( (ringCoordinates) => {
        ringCoordinates.forEach( (coordinate) => {
          let newCoordinate = [
            Math.abs(coordinate[0] - extent[0]) / widthResolution,
            Math.abs((Math.abs(coordinate[1] - extent[1]) / heightResolution) - 512)
          ];
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
