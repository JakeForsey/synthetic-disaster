import { Injectable } from '@angular/core';

import OlXYZ from 'ol/source/XYZ';
import OlTileLayer from 'ol/layer/Tile';

@Injectable({
  providedIn: 'root'
})
export class BackgroundService {

  private source: OlXYZ
  private layer: OlTileLayer

  constructor() {
    this.source = new OlXYZ({
      url: 'http://tile.osm.org/{z}/{x}/{y}.png'
    });

    this.layer = new OlTileLayer({
      source: this.source
    });
  }

  getLayer() {
    return this.layer;
  }
}
