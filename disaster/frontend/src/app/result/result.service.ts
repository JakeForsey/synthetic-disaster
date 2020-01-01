import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { HttpParams } from "@angular/common/http";

import { map } from 'rxjs/operators';
import { EditService } from 'src/app/map/edit.service';

@Injectable({
  providedIn: 'root'
})
export class ResultService {

  private editService: EditService
  scene: string

  constructor(
    private http: HttpClient,
    editService: EditService
    ) {
      this.editService = editService;
    }

  getImage() {
    let layout = this.editService.getLayout();
    layout["scene"] = this.scene;

    return this.http.get(
      "http://localhost:6001/generate",
      { responseType: "blob", params: new HttpParams().set('layout', JSON.stringify(layout)) },
    );
  }

}
