import { Injectable } from '@angular/core';

import OlView from 'ol/View';
import { fromLonLat } from 'ol/proj';


@Injectable({
  providedIn: 'root'
})
export class MapService {

  view: OlView;

  constructor() {
    this.view = new OlView({
      // GSI office
      center: fromLonLat([-1.707, 52.504411]),
      // 0.4 meters per pixel
      resolution: 0.4
    });
  }
}
